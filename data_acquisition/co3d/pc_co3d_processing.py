import numpy as np
import json
import gzip
from pathlib import Path
import imageio.v2 as imageio


def load_frame_annotations(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def project_depth(depth, obj_mask, depth_mask, fx, fy, cx, cy):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    valid = (depth > 0) & (obj_mask > 0) & (depth_mask > 0)

    if np.sum(valid) == 0:
        return np.zeros((0, 3))

    z = depth[valid]
    x = (xs[valid] - cx) * z / fx
    y = (ys[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def cam_to_world(points, R, T):
    return (R @ points.T).T + T


def write_ply(path, points):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def find_matching_file(folder, base_name):
    matches = list(folder.glob(base_name + "*"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for {base_name} in {folder}")
    return matches[0]


def process_object(obj_path: Path):
    print(f"\nGenerating GT point cloud for {obj_path.name}")

    ann_path = obj_path / "frame_annotations.jgz"
    annotations = load_frame_annotations(ann_path)

    print(type(annotations))
    print(len(annotations))
    print(annotations[0].keys())

    depths_dir = obj_path / "depths"
    masks_dir = obj_path / "masks"
    depth_masks_dir = obj_path / "depth_masks"

    all_points = []

    for i, frame in enumerate(annotations):

        img_name = Path(frame["image"]["path"]).name
        frame_id = img_name.replace(".jpg", "")  # frame000006

        depth_file = depths_dir / (img_name + ".geometric.png")
        obj_mask_file = masks_dir / (frame_id + ".png")
        depth_mask_file = depth_masks_dir / (frame_id + ".png")

        if not depth_file.exists():
            raise FileNotFoundError(depth_file)

        if not obj_mask_file.exists():
            raise FileNotFoundError(obj_mask_file)

        if not depth_mask_file.exists():
            raise FileNotFoundError(depth_mask_file)

        depth = imageio.imread(depth_file).astype(np.float32)
        obj_mask = imageio.imread(obj_mask_file)
        depth_mask = imageio.imread(depth_mask_file)

        if i == 0:
            print("Depth stats:", depth.dtype, depth.min(), depth.max())
            print("Obj mask stats:", obj_mask.dtype, obj_mask.min(), obj_mask.max())
            print("Depth mask stats:", depth_mask.dtype, depth_mask.min(), depth_mask.max())

        cam = frame["camera"]

        if "intrinsics" in cam:
            fx = cam["intrinsics"]["fx"]
            fy = cam["intrinsics"]["fy"]
            cx = cam["intrinsics"]["cx"]
            cy = cam["intrinsics"]["cy"]
        else:
            K = np.array(cam["K"])
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

        pts_cam = project_depth(depth, obj_mask, depth_mask, fx, fy, cx, cy)

        R = np.array(cam["R"])
        T = np.array(cam["T"])

        pts_world = cam_to_world(pts_cam, R, T)
        all_points.append(pts_world)

    all_points = np.concatenate(all_points, axis=0)

    print("Total points:", len(all_points))

    write_ply(obj_path / "gt_pointcloud.ply", all_points)
    print("Saved:", obj_path / "gt_pointcloud.ply")


if __name__ == "__main__":
    ROOT = Path("../data/co3D")

    for category in ROOT.iterdir():
        if category.is_dir():
            process_object(category)
            break  
    print("\nDone.")
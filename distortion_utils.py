from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

# VGGT util
from vggt.utils.geometry import unproject_depth_map_to_point_map


# =============================
# Pose (images.txt) -> extrinsic (world -> cam)
# =============================
def quat_to_R_hamilton(qw, qx, qy, qz) -> np.ndarray:
    qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 0:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )

def load_extrinsics_world_to_cam(images_txt: Path, frame_stem: str) -> np.ndarray:
    target_upper = f"{frame_stem}.JPG"
    target_lower = f"{frame_stem}.jpg"

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                if name.endswith(target_upper) or name.endswith(target_lower):
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    R = quat_to_R_hamilton(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                    return np.concatenate([R, t], axis=1)  # (3,4) world->cam
    raise FileNotFoundError(f"Frame {frame_stem} not found in {images_txt}")


# =============================
# THIN_PRISM_FISHEYE forward projection
# =============================
def thin_prism_fisheye_project(xd, yd, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params

    r = np.sqrt(xd * xd + yd * yd)
    theta = np.arctan(r)

    scale = np.where(r > 1e-12, theta / r, 1.0)
    ud = scale * xd
    vd = scale * yd

    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4

    tr = 1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8

    un = ud * tr + 2 * p1 * ud * vd + p2 * (th2 + 2 * ud * ud) + sx1 * th2
    vn = vd * tr + 2 * p2 * ud * vd + p1 * (th2 + 2 * vd * vd) + sy1 * th2

    u = fx * un + cx
    v = fy * vn + cy
    return u, v


def undistort_depth_to_pinhole(depth_dist, pinhole_K, out_hw, fish_params):
    H_u, W_u = out_hw
    H_d, W_d = depth_dist.shape

    fxu, fyu = float(pinhole_K[0, 0]), float(pinhole_K[1, 1])
    cxu, cyu = float(pinhole_K[0, 2]), float(pinhole_K[1, 2])

    uu, vv = np.meshgrid(
        np.arange(W_u, dtype=np.float32),
        np.arange(H_u, dtype=np.float32),
    )

    xd = (uu - cxu) / fxu
    yd = (vv - cyu) / fyu

    u_d, v_d = thin_prism_fisheye_project(xd, yd, fish_params)

    u_i = np.rint(u_d).astype(np.int32)
    v_i = np.rint(v_d).astype(np.int32)

    inside = (u_i >= 0) & (u_i < W_d) & (v_i >= 0) & (v_i < H_d)

    depth_und = np.full((H_u, W_u), np.inf, dtype=np.float32)
    depth_und[inside] = depth_dist[v_i[inside], u_i[inside]]
    return depth_und


# =============================
# COLMAP text helpers
# =============================
def find_camera_id_for_frame(images_txt: Path, frame_stem: str):
    target_upper = f"{frame_stem}.JPG"
    target_lower = f"{frame_stem}.jpg"

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                if name.endswith(target_upper) or name.endswith(target_lower):
                    cam_id = int(parts[8])
                    return cam_id, name
    raise FileNotFoundError(f"Frame {frame_stem} not found in {images_txt}")


def load_camera_block(cameras_txt: Path, camera_id: int):
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if int(parts[0]) == camera_id:
                model = parts[1]
                W = int(parts[2])
                H = int(parts[3])
                params = list(map(float, parts[4:]))
                return model, W, H, params
    raise FileNotFoundError(f"Camera id {camera_id} not found in {cameras_txt}")


def load_eth3d_depth_distorted(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "distorded_images" / "cameras.txt"
    images_txt = scene_path / "distorded_images" / "images.txt"
    depth_path = scene_path / "depths" / f"{frame_stem}.JPG"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, _ = load_camera_block(cameras_txt, cam_id)

    data = np.fromfile(str(depth_path), dtype=np.float32)
    if data.size != H * W:
        raise ValueError(
            f"Depth size mismatch for {depth_path.name}: got {data.size}, expected {H*W} (H={H}, W={W})."
        )
    return data.reshape((H, W)), cam_id, model


def load_fisheye_params(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "distorded_images" / "cameras.txt"
    images_txt = scene_path / "distorded_images" / "images.txt"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, params = load_camera_block(cameras_txt, cam_id)
    if model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Distorted camera model is {model}, expected THIN_PRISM_FISHEYE.")
    if len(params) < 12:
        raise ValueError(f"THIN_PRISM_FISHEYE expects 12 params, got {len(params)}.")
    return params[:12], (H, W), cam_id, model


def load_undistorted_pinhole_K(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "undistorded_images" / "cameras.txt"
    images_txt = scene_path / "undistorded_images" / "images.txt"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, params = load_camera_block(cameras_txt, cam_id)
    if model != "PINHOLE":
        raise ValueError(f"Undistorted camera model is {model}, expected PINHOLE.")

    fx, fy, cx, cy = params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, (H, W), cam_id, model


# =============================
# Resize helpers + viz
# =============================
def resize_depth_keep_inf(depth_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    valid = np.isfinite(depth_hw) & (depth_hw > 0)
    d = depth_hw.copy()
    d[~valid] = 0.0

    d_img = Image.fromarray(d.astype(np.float32), mode="F")
    v_img = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")

    d_r = np.array(d_img.resize((out_w, out_h), resample=Image.NEAREST), dtype=np.float32)
    v_r = np.array(v_img.resize((out_w, out_h), resample=Image.NEAREST), dtype=np.uint8) > 0

    d_r[~v_r] = np.inf
    return d_r


def scale_K_pinhole(K: np.ndarray, in_hw, out_hw) -> np.ndarray:
    H0, W0 = in_hw
    H1, W1 = out_hw
    sx = W1 / W0
    sy = H1 / H0
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def get_depth_pred_uint8(frame_stem: str, pred_scene_path: Path) -> np.ndarray:
    path = pred_scene_path / "depths" / f"{frame_stem}.jpg"
    return np.array(Image.open(path))


def get_image_518_distorted(frame_stem: str, scene_path: Path, out_h: int, out_w: int) -> np.ndarray:
    img_path = scene_path / "undistorded_images" / "images" / f"{frame_stem}.JPG"
    img = Image.open(img_path).convert("RGB")
    img_ = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.array(img_)


def show_triplet_side_by_side(img_518, pred_depth, gt_depth_518,
                              title_img="Image (518x518)",
                              title_pred="Pred (uint8)",
                              title_gt="GT depth undist (float32)"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(img_518)
    axes[0].set_title(title_img)
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_depth, cmap="gray")
    axes[1].set_title(title_pred)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    gt = gt_depth_518.astype(np.float32).copy()
    valid = np.isfinite(gt) & (gt > 0)
    if valid.any():
        vmax = np.percentile(gt[valid], 99)
        gt_vis = np.clip(gt, 0, vmax)
        gt_vis[~valid] = 0.0
    else:
        vmax = 1.0
        gt_vis = np.zeros_like(gt)

    im2 = axes[2].imshow(gt_vis, cmap="gray")
    axes[2].set_title(f"{title_gt} (clip p99={vmax:.3g})")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def build_undistorted_depth_and_K(scene_path: Path, frame_stem: str):
    depth_dist, _, _ = load_eth3d_depth_distorted(scene_path, frame_stem)
    fish_params, _, _, _ = load_fisheye_params(scene_path, frame_stem)
    K_und, und_hw, _, _ = load_undistorted_pinhole_K(scene_path, frame_stem)
    depth_und = undistort_depth_to_pinhole(depth_dist, K_und, und_hw, fish_params)
    return depth_und, K_und, und_hw

def pointmap_to_pointcloud(pointmap_world: np.ndarray, depth_valid: np.ndarray, eps=1e-8):
    """
    pointmap_world: (H,W,3) float
    depth_valid: (H,W) depth utilisée pour créer la pointmap (avec 0 pour invalid)
    Retour: pts (N,3)
    """
    valid = (depth_valid > eps) & np.isfinite(pointmap_world[..., 0]) & np.isfinite(pointmap_world[..., 1]) & np.isfinite(pointmap_world[..., 2])
    pts = pointmap_world[valid].reshape(-1, 3).astype(np.float32)
    return pts


def get_pm_pred(frame, pred_scene_path) :
    path = pred_scene_path / "pointmaps" / f"{frame}.npy"
    pm  = np.load(path).astype(np.float32) 
    return pm

def get_maps_depths(frame, gt_scene_path, pred_scene_path) :
    
    # Pred depth (uint8 image)
    pred_depth = get_depth_pred_uint8(frame, pred_scene_path)
    pred_pm = get_pm_pred(frame, pred_scene_path)
    out_h, out_w = pred_depth.shape[:2]  # 518,518

    # 1) Build undistorted GT depth + K (native undistorted resolution)
    depth_und, K_und, und_hw = build_undistorted_depth_and_K(gt_scene_path, frame)

    # 2) Resize GT depth to 518 and scale intrinsics accordingly
    depth_und_518 = resize_depth_keep_inf(depth_und, out_h, out_w)
    K_518 = scale_K_pinhole(K_und, und_hw, (out_h, out_w))

    # Replace inf/nan by 0 before unproject.
    depth_und_518_for_unproj = depth_und_518.copy()
    depth_und_518_for_unproj[~np.isfinite(depth_und_518_for_unproj)] = 0.0

    # 3) Load GT extrinsics (world -> cam) for the UNDISTORTED image
    images_txt_und = gt_scene_path / "undistorded_images" / "images.txt"
    extr_w2c = load_extrinsics_world_to_cam(images_txt_und, frame)  # (3,4)

    # 4) Unproject to WORLD points (S=1)
    pm_gt_world = unproject_depth_map_to_point_map(
        depth_und_518_for_unproj[None, ..., None],  # (1,H,W,1)
        extr_w2c[None, ...],                        # (1,3,4) world->cam
        K_518[None, ...],                           # (1,3,3)
    )

    return pm_gt_world, depth_und_518, pred_pm, pred_depth


def show_pointmap_o3d(pointmap, depth=None, eps=1e-8, max_points=None):
    """
    pointmap: (H,W,3) ou (1,H,W,3)
    depth: (H,W) optionnel (0/inf = invalide). Si fourni, sert à filtrer.
    """
    pm = np.asarray(pointmap)
    if pm.ndim == 4:
        pm = pm[0]
    pm = pm.astype(np.float32)

    valid = np.isfinite(pm[..., 0]) & np.isfinite(pm[..., 1]) & np.isfinite(pm[..., 2])
    if depth is not None:
        d = np.asarray(depth)
        valid &= np.isfinite(d) & (d > eps)

    pts = pm[valid].reshape(-1, 3)

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    dataset_path = Path("data/eth3D/")
    preds_path = Path("outputs/eth3D_local/")

    scene = "office"
    frame = "DSC_0240"

    gt_scene_path = dataset_path / scene
    pred_scene_path = preds_path / scene

    gt_pm, gt_depth, pred_pm, pred_depth = get_gt(frame, pred_scene_path)
    

import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import gc


def parse_all_mlp_matrices(mlp_path):
    tree = ET.parse(mlp_path)
    root = tree.getroot()

    matrices = {}

    for mesh in root.iter("MLMesh"):
        label = mesh.attrib.get("label")
        matrix_elem = mesh.find("MLMatrix44")

        if matrix_elem is not None:
            values = list(map(float, matrix_elem.text.strip().split()))
            matrix = np.array(values).reshape(4, 4)
            matrices[label] = matrix

    if len(matrices) == 0:
        raise ValueError("No matrices found in MLP file")

    return matrices


def merge_scans(scene_root_path, voxel_size=0.005):

    scene_root_path = Path(scene_root_path)
    scan_folder = scene_root_path / "dslr_scan_eval"
    mlp_path = scan_folder / "scan_alignment.mlp"

    if not mlp_path.exists():
        print(f"[SKIP] Missing MLP in {scan_folder}")
        return

    matrices = parse_all_mlp_matrices(str(mlp_path))

    reference_label = sorted(matrices.keys())[0]
    T_ref = matrices[reference_label]

    merged_cloud = o3d.geometry.PointCloud()

    for label, T_i in matrices.items():

        ply_path = scan_folder / label
        if not ply_path.exists():
            continue

        pcd = o3d.io.read_point_cloud(str(ply_path))

        T_relative = np.linalg.inv(T_ref) @ T_i
        pcd.transform(T_relative)

        # Downsample immediately
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)

        merged_cloud += pcd

        # Free memory explicitly
        del pcd
        gc.collect()

    # Optional final cleanup downsample
    if voxel_size is not None:
        merged_cloud = merged_cloud.voxel_down_sample(voxel_size)

    output_path = scene_root_path / "scan_merged.ply"
    o3d.io.write_point_cloud(str(output_path), merged_cloud, write_ascii=False)

    del merged_cloud
    gc.collect()


def process_all_scenes(root_folder, voxel_size=0.005):

    root_folder = Path(root_folder)
    scenes = [d for d in root_folder.iterdir() if d.is_dir()]

    for scene_dir in tqdm(scenes, desc="Merging ETH3D scans"):
        merge_scans(scene_dir, voxel_size=voxel_size)


if __name__ == "__main__":
    root_dataset_path = "../data/eth3D_raw/multi_view_training_dslr_scan_eval"
    process_all_scenes(root_dataset_path, voxel_size=0.002)
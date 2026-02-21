import shutil
from pathlib import Path
from tqdm import tqdm

def safe_copy(src, dst):
    if src.is_dir():
        if not dst.exists():
            shutil.copytree(src, dst)
    else:
        if not dst.exists():
            shutil.copy2(src, dst)

def build_eth3d_clean(
    root_occlusion,
    root_scan_eval,
    root_undistorted,
    output_root,
):
    root_occlusion = Path(root_occlusion)
    root_scan_eval = Path(root_scan_eval)
    root_undistorted = Path(root_undistorted)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    scenes = sorted([d.name for d in root_undistorted.iterdir() if d.is_dir()])

    for scene in tqdm(scenes, desc="Building eth3d_clean"):
        scene_out = output_root / scene
        scene_out.mkdir(exist_ok=True)

        scan_path = root_scan_eval / scene / "scan_merged.ply"
        if scan_path.exists():
            safe_copy(scan_path, scene_out / "scan_merged.ply")
        else:
            print(f"[WARN] Missing scan_merged for {scene}")

        images_src = root_undistorted / scene / "images"
        if images_src.exists():
            safe_copy(images_src, scene_out / "images")
        else:
            print(f"[WARN] Missing images for {scene}")

        calib_src = root_undistorted / scene / "dslr_calibration_undistorted"

        cameras_src = calib_src / "cameras.txt"
        images_txt_src = calib_src / "images.txt"

        if cameras_src.exists():
            safe_copy(cameras_src, scene_out / "cameras.txt")
        else:
            print(f"[WARN] Missing cameras.txt for {scene}")

        if images_txt_src.exists():
            safe_copy(images_txt_src, scene_out / "images.txt")
        else:
            print(f"[WARN] Missing images.txt for {scene}")

        masks_src = root_occlusion / scene / "masks_for_images"
        if masks_src.exists():
            safe_copy(masks_src, scene_out / "masks_for_images")
        else:
            print(f"[INFO] No masks for {scene} (optional)")

    print("\nDone building eth3d_clean.")


if __name__ == "__main__":

    build_eth3d_clean(
        root_occlusion="../data/eth3D_raw/multi_view_training_dslr_occlusion",
        root_scan_eval="../data/eth3D_raw/multi_view_training_dslr_scan_eval",
        root_undistorted="../data/eth3D_raw/multi_view_training_dslr_undistorted",
        output_root="../data/eth3d_clean",
    )
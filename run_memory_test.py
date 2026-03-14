import os
import time
import random
import torch
import numpy as np
from pathlib import Path
import argparse

from vggt.utils.load_fn import load_and_preprocess_images_square
from scripts.utils import preprocess_eth3d_masks, get_device_settings, synchronize_device, empty_gpu_cache
from scripts.vggt_ops import load_vggt_model, run_VGGT

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Memory testing for VGGT")
    parser.add_argument("--checkpoint", type=str, default="models/model.pt")
    parser.add_argument("--dataset_path", type=str, default="data/eth3D/")
    parser.add_argument("--scene_name", type=str, default="courtyard", help="Name of the scene to test")
    
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    vggt_fixed_resolution = 518
    img_load_resolution = 1024 

    checkpoint_path = args.checkpoint
    dataset_path = Path(args.dataset_path)
    scene_name = args.scene_name
    scene = dataset_path / scene_name

    n_frames_list = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20]

    device, dtype = get_device_settings()
    print(f"Device: {device}")

    model = load_vggt_model(checkpoint_path, device)
    print("Model loaded")

    image_dir = scene / "undistorded_images/images"

    image_path_list = sorted(
        [str(f) for f in image_dir.iterdir() if not f.name.startswith(".")]
    )

    if len(image_path_list) == 0:
        raise RuntimeError("No images found")

    random.shuffle(image_path_list)

    print(f"\nTesting scene: {scene_name}")
    print("-" * 40)

    for n_frames in n_frames_list:

        if n_frames > len(image_path_list):
            break

        sampled_paths = image_path_list[:n_frames]

        # Load masks
        masks_dir = scene / "masks_for_images"
        mask_path_list = [masks_dir / f"{Path(p).stem}.png" for p in sampled_paths]

        masks = preprocess_eth3d_masks(
            mask_path_list,
            target_size=vggt_fixed_resolution
        )

        masks = (masks != 2)

        # Load images
        images, _ = load_and_preprocess_images_square(
            sampled_paths,
            img_load_resolution
        )

        images = images.to(device)

        
        empty_gpu_cache(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        synchronize_device(device)

        start_time = time.time()
        
        with torch.no_grad():
            extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
                model,
                images,
                vggt_fixed_resolution
            )

        synchronize_device(device)

        end_time = time.time()

        
        if device.type == "cuda":
            peak_mem_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem_mb = peak_mem_bytes / (1024 ** 2)
        else:
            peak_mem_mb = 0.0

        duration = end_time - start_time

        print(
            f"N={n_frames:2d} frames | "
            f"time = {duration:.2f}s | "
            f"GPU memory = {peak_mem_mb:.0f} MB"
        )
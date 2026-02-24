import os
import trimesh
import time
import numpy as np
from pathlib import Path
import random
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from utils import preprocess_eth3d_masks, get_device_settings, synchronize_device, get_gpu_memory_usage, empty_gpu_cache
from vggt_ops import load_vggt_model, run_VGGT, post_processing_pc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "models/model.pt"
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    n_frames = 7
    depth_threshold = 1.0
    dataset_path = Path("data/eth3D/")
    outputs_path = Path("outputs/eth3D_local/")
    
    device, dtype = get_device_settings()
    print(f"Device: {device}")

    # Load Model
    model = load_vggt_model(checkpoint_path, device)
    print(f"Model successfully loaded on {device}")

    inference_durations = []
    mem_useds = []

    for scene in dataset_path.iterdir() :
        if scene.name.startswith('.'):
            continue
            
        #### FOR THE TEST
        target_scene_name = "office"
        scene = dataset_path / target_scene_name

        print(f"\nProcessing scene: {scene.name}")

        # Load and Preprocess Images
        image_dir = scene / "undistorded_images"
        image_path_list = [str(f) for f in image_dir.iterdir() if not f.name.startswith('.')]
        if not image_path_list:
            print(f"No images found in {image_dir}, skipping.")
            continue

        num_to_sample = min(len(image_path_list), n_frames) 
        sampled_paths = random.sample(image_path_list, num_to_sample)
        
        masks_dir = scene / "masks_for_images"
        mask_path_list = [masks_dir / f"{Path(p).stem}.png" for p in sampled_paths]
        masks = preprocess_eth3d_masks(mask_path_list, target_size=vggt_fixed_resolution)
        masks = (masks != 2)

        images, _ = load_and_preprocess_images_square(sampled_paths, img_load_resolution)
        images = images.to(device)

        # --- Start Inference Tracking ---
        empty_gpu_cache(device)
        mem_before = get_gpu_memory_usage(device)
        start_time = time.time()

        # Inference
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)

        # --- End Inference Tracking ---
        synchronize_device(device)
        end_time = time.time()
        mem_after = get_gpu_memory_usage(device)
        inference_durations.append(end_time - start_time)
        mem_useds.append(mem_after - mem_before)

        # Post-processing
        pointmaps = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        print(f"Pointmaps shape: {pointmaps.shape}")

        points_3d, points_rgb, depths = post_processing_pc(pointmaps.copy(), images, depth_map, vggt_fixed_resolution, masks, depth_conf, depth_threshold)

        # Export
        res_dir = outputs_path / scene.name
        pm_dir = res_dir / "pointmaps"
        conf_dir = res_dir / "confidence"
        depth_dir = res_dir / "depths"
        extr_dir = res_dir / "extrinsic"
        intr_dir = res_dir / "intrinsic"
        
        for d in [pm_dir, conf_dir, depth_dir, extr_dir, intr_dir]:
            d.mkdir(parents=True, exist_ok=True)

        full_path = res_dir / f"scan.ply"    
        trimesh.PointCloud(points_3d, colors=points_rgb).export(full_path) 

        for i, img_path in enumerate(sampled_paths):
            img_name = Path(img_path).stem

            np.save(pm_dir / f"{img_name}.npy", pointmaps[i].astype(np.float32))
            np.save(conf_dir / f"{img_name}.npy", depth_conf[i].astype(np.float32))
            np.save(extr_dir / f"{img_name}.npy", extrinsic[i])
            np.save(intr_dir / f"{img_name}.npy", intrinsic[i])

            d_max = depths[i].max()
            if d_max > 0:
                d_img = (depths[i] / d_max * 255).astype(np.uint8)
            else:
                d_img = np.zeros_like(depths[i], dtype=np.uint8)
            Image.fromarray(d_img).save(res_dir / "depths" / f"{img_name}.jpg")

        break  

    mean_duration = np.mean(inference_durations)
    mean_memory = np.mean(mem_useds)

    print(f"Mean duration : {mean_duration:.02f}s")
    print(f"Mean memory usage : {mean_memory:.02f} MB")
    
        

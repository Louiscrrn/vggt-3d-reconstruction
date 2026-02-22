import glob
import os
import trimesh
import time
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from utils import get_device_settings, synchronize_device, get_gpu_memory_usage, empty_gpu_cache
from vggt_ops import load_vggt_model, run_VGGT, post_processing_pc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "models/model.pt"
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    n_frames = 3
    path_list = "data/examples/llff_fern"
    scene_dir = "outputs/"
    dump_name = "flower"
    
    device, dtype = get_device_settings()
    print(f"Device: {device}")

    # Load Model
    model = load_vggt_model(checkpoint_path, device)
    print(f"Model successfully loaded on {device}")

    # Load and Preprocess Images
    image_dir = os.path.join(path_list, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    
    images = images.to(device)[:n_frames]
    original_coords = original_coords.to(device)
    print(f"Loaded {images.shape} images from {image_dir}")

    # --- Start Inference Tracking ---
    empty_gpu_cache(device)
    mem_before = get_gpu_memory_usage(device)
    
    start_time = time.time()

    # Inference
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    
    # Crucial for accurate timing on GPU
    synchronize_device(device)
    
    end_time = time.time()
    mem_after = get_gpu_memory_usage(device)
    # --- End Inference Tracking ---

    # Post-processing
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    points_3d, points_rgb = post_processing_pc(points_3d, images, vggt_fixed_resolution, depth_conf)
   
    # Export
    os.makedirs(scene_dir, exist_ok=True)
    filename = f"{dump_name}_n{images.shape[0]}_{device.type}.ply"    
    full_path = os.path.join(scene_dir, filename)
    trimesh.PointCloud(points_3d, colors=points_rgb).export(full_path)
    
    # Results Printing
    inference_duration = end_time - start_time
    mem_used = mem_after - mem_before
    
    print("-" * 30)
    print(f"Inference Results:")
    print(f"Execution time: {inference_duration:.4f} s")
    if device.type in ["cuda", "mps"]:
        print(f"Allocated GPU memory: {mem_used:.2f} MB")
    print(f"File saved in: {full_path}")
    print("-" * 30)

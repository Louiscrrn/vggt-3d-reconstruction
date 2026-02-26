import torch
from PIL import Image
import numpy as np
from pathlib import Path

def load_eth3d_depth(path):
    depth = np.fromfile(path, dtype=np.float32)
    size = depth.shape[0]
    return 0

def get_eth3d_mask(mask_path, target_size=518) :
    mask = Image.open(mask_path).convert("L")
    width, height = mask.size
        
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
        
    square_mask = Image.new("L", (max_dim, max_dim), 0)
    square_mask.paste(mask, (left, top))
        
    square_mask = square_mask.resize(
    (target_size, target_size),
            Image.Resampling.NEAREST
    )
        
    mask_array = np.array(square_mask, dtype=np.uint8)
    return mask_array


def preprocess_eth3d_masks(mask_path_list, target_size=518):
    processed_masks = []
    
    for mask_path in mask_path_list:
        if not mask_path.exists():
            print(f"Warning: Mask not found at {mask_path}")
            processed_masks.append(np.zeros((target_size, target_size), dtype=np.uint8))
            continue
       
        mask = Image.open(mask_path).convert("L")

        width, height = mask.size
        
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        
        square_mask = Image.new("L", (max_dim, max_dim), 0)
        square_mask.paste(mask, (left, top))
        
        square_mask = square_mask.resize(
            (target_size, target_size),
            Image.Resampling.NEAREST
        )
        
        mask_array = np.array(square_mask, dtype=np.uint8)
        processed_masks.append(mask_array)
        
    return np.stack(processed_masks)

def get_device_settings():
    """
    Returns:
        device (torch.device)
        dtype (torch.dtype) preferred for inference
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16  # best speed on GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # float16 unstable on MPS
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def get_autocast_args(device):
    """
    Returns kwargs compatible with:
        with torch.amp.autocast(**args):
    """

    if device.type == "cuda":
        return {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": True,
        }

    elif device.type == "mps":
        return {
            "device_type": "cpu",
            "enabled": False,
        }

    else:
        return {
            "device_type": "cpu",
            "enabled": False,
        }
    

def synchronize_device(device):
    """Synchronizes the device to ensure timing measurements are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def get_gpu_memory_usage(device):
    """Returns the currently allocated GPU memory in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0

def empty_gpu_cache(device):
    """Clears the GPU cache."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

def qvec2rotmat(qvec):
    """Convertit un quaternion [qw, qx, qy, qz] en matrice de rotation 3x3."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def load_eth3d_pose_logic(image_name, scene_path):
    # 1. Charger Cameras
    cams = {}
    with open(scene_path / "cameras.txt", "r") as f:
        for line in f:
            if line.startswith("#"): continue
            # ID, MODEL, WIDTH, HEIGHT, PARAMS...
            els = line.split()
            cams[int(els[0])] = {
                "W": int(els[2]), 
                "H": int(els[3]), 
                "K": np.array([[float(els[4]), 0, float(els[6])],
                               [0, float(els[5]), float(els[7])],
                               [0, 0, 1]])
            }

    # 2. Charger Images
    with open(scene_path / "images.txt", "r") as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):
            els = lines[i].split()
            if image_name in els[9]:
                qw, qx, qy, qz = map(float, els[1:5])
                tx, ty, tz = map(float, els[5:8])
                cam_id = int(els[8])
                
                R = qvec2rotmat([qw, qx, qy, qz])
                t = np.array([tx, ty, tz])
                
                # On renvoie tout, résolution incluse
                return R, t, cams[cam_id]["K"], cams[cam_id]["W"], cams[cam_id]["H"]
    
    return None
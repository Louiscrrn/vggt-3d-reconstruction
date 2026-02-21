import torch

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
            "dtype": torch.float16,
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
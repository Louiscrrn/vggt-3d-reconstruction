import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_image_camera_id(images_txt, image_name):
    with open(images_txt) as f:
        for line in f:
            if image_name in line:
                parts = line.split()
                return int(parts[-2]) 
    raise ValueError("Image non trouvée dans images.txt")


def get_camera_resolution(cameras_txt, camera_id):
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith(str(camera_id) + " "):
                parts = line.split()
                width = int(parts[2])
                height = int(parts[3])
                return height, width
    raise ValueError("Camera ID non trouvé")


def load_depth(depth_path, images_txt, cameras_txt):
    image_name = depth_path.name
    
    camera_id = get_image_camera_id(images_txt, image_name)
    print(f"Camera ID : {camera_id}")
    height, width = get_camera_resolution(cameras_txt, camera_id)
    print(f"H, W : {height}, {width}")

    depth = np.fromfile(depth_path, dtype=np.float32)
    print(depth.size)
    print(height * width)
    if depth.size != height * width:
        raise ValueError("Mismatch taille depth / résolution caméra")
    
    depth = depth.reshape((height, width))
    depth[~np.isfinite(depth)] = 0.0
    
    return depth

# --- TEST D'AFFICHAGE ---
scene_path = Path("data/eth3D/electro")

depth_dir = scene_path / "depths"
images_txt = scene_path / "images.txt"
cameras_txt = scene_path / "cameras.txt"

# On prend la première depth disponible
depth_file = depth_dir / "DSC_9278.JPG"

print("Testing:", depth_file.name)

depth = load_depth(depth_file, images_txt, cameras_txt)

# Visualisation propre
valid = depth[depth > 0]

vmin = np.percentile(valid, 2)
vmax = np.percentile(valid, 98)

plt.figure(figsize=(10, 6))
plt.imshow(depth, cmap="magma", vmin=vmin, vmax=vmax)
plt.colorbar(label="Profondeur (m)")
plt.title(depth_file.name)
plt.axis("off")
plt.show()
import torch
import torch.nn.functional as F
import glob
import os
import copy
import numpy as np
import trimesh
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

from vggt.models.vggt import VGGT
from vggt.utils.utils_fn import *

def load_vggt_model(checkpoint_path, device):
    model = VGGT()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        autocast_args = get_autocast_args(images.device)
        with torch.amp.autocast(**autocast_args):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction

def colmap_conv(points_3d, vggt_fixed_resolution, depth_conf, extrinsic, intrinsic) :
    
    conf_thres_value = 5.0
    max_points_for_colmap = 100000  # randomly sample 3D points
    shared_camera = False  # in the feedforward manner, we do not support shared camera
    camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= conf_thres_value
    # at most writing 100000 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
    )

    reconstruction_resolution = vggt_fixed_resolution

    return reconstruction, points_rgb

def post_processing_pc(points_3d, images, vggt_fixed_resolution, depth_conf) :
    points_rgb = F.interpolate(images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear")
    colors = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    mask = depth_conf > 5.0 
    valid_points = points_3d[mask]
    valid_colors = colors[mask]

    return valid_points, valid_colors

if __name__ == "__main__":

    path = "models/vanilla.pt"
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    batch = 10
    path_list = "examples/llff_fern/"
    scene_dir = "outputs/"
    dump_name = "flower"
    
    device, dtype = get_device_settings()
    print(f"Device : {device}")

    model = load_vggt_model(path, device)
    print("Model sucessfully loaded", device)

    image_dir = os.path.join(path_list, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    images = images[:batch]
    original_coords = original_coords.to(device)
    print(f"Loaded {images.shape} images from {image_dir}")

    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    points_3d, points_rgb = post_processing_pc(points_3d, images, vggt_fixed_resolution, depth_conf)
   
    os.makedirs(scene_dir, exist_ok=True)
    filepath = scene_dir + f"{dump_name}_points.ply"
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(scene_dir, filepath))
    print(f"File saved in : {filepath}")
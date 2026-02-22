import torch
import torch.nn.functional as F
import numpy as np

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

from utils import get_autocast_args

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

def post_processing_pc(points_3d, images, depth_map, vggt_fixed_resolution, depth_conf, depth_threshold=5.0) :
    
    points_rgb = F.interpolate(images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear")
    colors = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    
    conf_mask = (depth_conf > depth_threshold)
    mask = conf_mask 

    valid_depths = depth_map.squeeze(-1).copy()
    valid_depths[~mask] = 0
    valid_points = points_3d[mask]
    valid_colors = colors[mask]

    print(f"Survivants Finaux: {np.sum(mask)} points")

    return valid_points, valid_colors, valid_depths

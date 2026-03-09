from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import random
from scipy.ndimage import maximum_filter
import cv2

# VGGT util
from vggt.utils.geometry import unproject_depth_map_to_point_map


# =============================
# Pose (images.txt) -> extrinsic (world -> cam)
# =============================
def quat_to_R_hamilton(qw, qx, qy, qz) -> np.ndarray:
    qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 0:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )

def load_extrinsics_world_to_cam(images_txt: Path, frame_stem: str) -> np.ndarray:
    target_upper = f"{frame_stem}.JPG"
    target_lower = f"{frame_stem}.jpg"

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                if name.endswith(target_upper) or name.endswith(target_lower):
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    R = quat_to_R_hamilton(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                    return np.concatenate([R, t], axis=1)  # (3,4) world->cam
    raise FileNotFoundError(f"Frame {frame_stem} not found in {images_txt}")


# =============================
# THIN_PRISM_FISHEYE forward projection
# =============================
def thin_prism_fisheye_project(xd, yd, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params

    r = np.sqrt(xd * xd + yd * yd)
    theta = np.arctan(r)

    scale = np.where(r > 1e-12, theta / r, 1.0)
    ud = scale * xd
    vd = scale * yd

    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4

    tr = 1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8

    un = ud * tr + 2 * p1 * ud * vd + p2 * (th2 + 2 * ud * ud) + sx1 * th2
    vn = vd * tr + 2 * p2 * ud * vd + p1 * (th2 + 2 * vd * vd) + sy1 * th2

    u = fx * un + cx
    v = fy * vn + cy
    return u, v


def undistort_depth_to_pinhole(depth_dist, pinhole_K, out_hw, fish_params):
    H_u, W_u = out_hw
    H_d, W_d = depth_dist.shape

    fxu, fyu = float(pinhole_K[0, 0]), float(pinhole_K[1, 1])
    cxu, cyu = float(pinhole_K[0, 2]), float(pinhole_K[1, 2])

    uu, vv = np.meshgrid(
        np.arange(W_u, dtype=np.float32),
        np.arange(H_u, dtype=np.float32),
    )

    xd = (uu - cxu) / fxu
    yd = (vv - cyu) / fyu

    u_d, v_d = thin_prism_fisheye_project(xd, yd, fish_params)

    u_i = np.rint(u_d).astype(np.int32)
    v_i = np.rint(v_d).astype(np.int32)

    inside = (u_i >= 0) & (u_i < W_d) & (v_i >= 0) & (v_i < H_d)

    depth_und = np.full((H_u, W_u), np.inf, dtype=np.float32)
    depth_und[inside] = depth_dist[v_i[inside], u_i[inside]]
    return depth_und


# =============================
# COLMAP text helpers
# =============================
def find_camera_id_for_frame(images_txt: Path, frame_stem: str):
    target_upper = f"{frame_stem}.JPG"
    target_lower = f"{frame_stem}.jpg"

    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[9]
                if name.endswith(target_upper) or name.endswith(target_lower):
                    cam_id = int(parts[8])
                    return cam_id, name
    raise FileNotFoundError(f"Frame {frame_stem} not found in {images_txt}")


def load_camera_block(cameras_txt: Path, camera_id: int):
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if int(parts[0]) == camera_id:
                model = parts[1]
                W = int(parts[2])
                H = int(parts[3])
                params = list(map(float, parts[4:]))
                return model, W, H, params
    raise FileNotFoundError(f"Camera id {camera_id} not found in {cameras_txt}")



def load_fisheye_params(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "distorded_images" / "cameras.txt"
    images_txt = scene_path / "distorded_images" / "images.txt"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, params = load_camera_block(cameras_txt, cam_id)
    if model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Distorted camera model is {model}, expected THIN_PRISM_FISHEYE.")
    if len(params) < 12:
        raise ValueError(f"THIN_PRISM_FISHEYE expects 12 params, got {len(params)}.")
    return params[:12], (H, W), cam_id, model


def load_undistorted_pinhole_K(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "undistorded_images" / "cameras.txt"
    images_txt = scene_path / "undistorded_images" / "images.txt"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, params = load_camera_block(cameras_txt, cam_id)
    if model != "PINHOLE":
        raise ValueError(f"Undistorted camera model is {model}, expected PINHOLE.")

    fx, fy, cx, cy = params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, (H, W), cam_id, model


# =============================
# Resize helpers + viz
# =============================
def resize_depth_keep_inf(depth_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    valid = np.isfinite(depth_hw) & (depth_hw > 0)
    d = depth_hw.copy()
    d[~valid] = 0.0

    d_img = Image.fromarray(d.astype(np.float32), mode="F")
    v_img = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")

    d_r = np.array(d_img.resize((out_w, out_h), resample=Image.NEAREST), dtype=np.float32)
    v_r = np.array(v_img.resize((out_w, out_h), resample=Image.NEAREST), dtype=np.uint8) > 0

    d_r[~v_r] = np.inf
    return d_r


def scale_K_pinhole(K: np.ndarray, in_hw, out_hw) -> np.ndarray:
    H0, W0 = in_hw
    H1, W1 = out_hw
    sx = W1 / W0
    sy = H1 / H0
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def show_triplet_side_by_side(img_518, pred_depth, gt_depth_518,
                              title_img="Image (518x518)",
                              title_pred="Pred (uint8)",
                              title_gt="GT depth undist (float32)"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(img_518)
    axes[0].set_title(title_img)
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_depth, cmap="gray")
    axes[1].set_title(title_pred)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    gt = gt_depth_518.astype(np.float32).copy()
    valid = np.isfinite(gt) & (gt > 0)
    if valid.any():
        vmax = np.percentile(gt[valid], 99)
        gt_vis = np.clip(gt, 0, vmax)
        gt_vis[~valid] = 0.0
    else:
        vmax = 1.0
        gt_vis = np.zeros_like(gt)

    im2 = axes[2].imshow(gt_vis, cmap="gray")
    axes[2].set_title(f"{title_gt} (clip p99={vmax:.3g})")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



def pointmap_to_pointcloud(pointmap_world: np.ndarray, depth_valid: np.ndarray, eps=1e-8):
    """
    pointmap_world: (H,W,3) float
    depth_valid: (H,W) depth utilisée pour créer la pointmap (avec 0 pour invalid)
    Retour: pts (N,3)
    """
    valid = (depth_valid > eps) & np.isfinite(pointmap_world[..., 0]) & np.isfinite(pointmap_world[..., 1]) & np.isfinite(pointmap_world[..., 2])
    pts = pointmap_world[valid].reshape(-1, 3).astype(np.float32)
    return pts


def get_pm_depths(frame, gt_scene_path, pred_scene_path) :
    
    # Pred depth (uint8 image)
    pred_depth_metric, pred_depth_vis = get_depth_pre(frame, pred_scene_path)
    pred_pm = get_pm_pred(frame, pred_scene_path)
    out_h, out_w = pred_depth_metric.shape[:2] 

    # GT depth
    gt_depth_metric, gt_depth_vis, K_518 = get_depth_gt(frame, gt_scene_path, "undistorded_images", out_h, out_w)

    # GT extrinsics (world -> cam) for the UNDISTORTED image
    images_txt_und = gt_scene_path / "undistorded_images" / "images.txt"
    extr_w2c = load_extrinsics_world_to_cam(images_txt_und, frame)  

    depth_for_unproj = gt_depth_metric.copy()
    depth_for_unproj[~np.isfinite(depth_for_unproj)] = 0

    # Unproject to WORLD points (S=1)
    gt_pm = unproject_depth_map_to_point_map(
        depth_for_unproj[None, ..., None],  
        extr_w2c[None, ...],                       
        K_518[None, ...],                       
    )

    return gt_pm, gt_depth_vis, pred_pm, pred_depth_vis


def show_pointmap_o3d(pointmap, depth=None, eps=1e-8, max_points=None):
    """
    pointmap: (H,W,3) ou (1,H,W,3)
    depth: (H,W) optionnel (0/inf = invalide). Si fourni, sert à filtrer.
    """
    pm = np.asarray(pointmap)
    if pm.ndim == 4:
        pm = pm[0]
    pm = pm.astype(np.float32)

    valid = np.isfinite(pm[..., 0]) & np.isfinite(pm[..., 1]) & np.isfinite(pm[..., 2])
    if depth is not None:
        d = np.asarray(depth)
        valid &= np.isfinite(d) & (d > eps)

    pts = pm[valid].reshape(-1, 3)

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


def get_pm_pred(frame, pred_scene_path) :
    path = pred_scene_path / "pointmaps" / f"{frame}.npy"
    pm  = np.load(path).astype(np.float32) 
    return pm

def downsample_depth_preserving(depth_hw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Downsample une depth sparse en préservant les valeurs valides.
    Stratégie : dilation morphologique AVANT resize nearest-neighbor.
    """
    depth = depth_hw.copy()
    valid = np.isfinite(depth) & (depth > 0)
    
    # Calcul du facteur de downscale
    scale_h = depth_hw.shape[0] / target_h
    scale_w = depth_hw.shape[1] / target_w
    scale = max(scale_h, scale_w)
    
    # Taille du kernel = facteur de downscale (arrondi impair)
    k = int(scale)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    
    #print(f"  Depth originale: {depth_hw.shape}, target: ({target_h},{target_w}), kernel dilation: {k}x{k}")
    
    # Remplacer invalides par 0 pour la dilation
    depth_for_dilation = depth.copy()
    depth_for_dilation[~valid] = 0.0
    
    # Dilation : chaque pixel valide "s'étale" sur ses voisins
    depth_dilated = maximum_filter(depth_for_dilation, size=k)
    
    # Mask de validité dilaté aussi
    valid_dilated = maximum_filter(valid.astype(np.uint8), size=k).astype(bool)
    
    depth_dilated[~valid_dilated] = np.inf  # remettre invalide là où il n'y a vraiment rien
    
    # Resize avec nearest-neighbor (les valeurs sont maintenant "denses")
    depth_resized = cv2.resize(
        depth_dilated, 
        (target_w, target_h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    return depth_resized

def load_depth_distorded(scene_path: Path, frame_stem: str):
    cameras_txt = scene_path / "distorded_images" / "cameras.txt"
    images_txt = scene_path / "distorded_images" / "images.txt"
    depth_path = scene_path / "depths" / f"{frame_stem}.JPG"

    cam_id, _ = find_camera_id_for_frame(images_txt, frame_stem)
    model, W, H, _ = load_camera_block(cameras_txt, cam_id)

    data = np.fromfile(str(depth_path), dtype=np.float32)
    if data.size != H * W:
        raise ValueError(
            f"Depth size mismatch for {depth_path.name}: got {data.size}, expected {H*W} (H={H}, W={W})."
        )
    return data.reshape((H, W)), cam_id, model

def process_depth(depth):
    depth = depth.copy()
    depth[~np.isfinite(depth)] = np.nan
    valid_pixels = depth[np.isfinite(depth) & (depth > 0)]
    vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
    depth_vis = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    return depth_vis

def get_depth_gt(frame, gt_scene_path, im_type="distorded_images" ,out_h=518, out_w=518) :
    depth_dist, _, _ = load_depth_distorded(gt_scene_path, frame)

    if im_type == "distorded_images" :
        depth_metric = downsample_depth_preserving(depth_dist, out_h, out_w)
        depth_vis = process_depth(depth_metric)
        return depth_metric, depth_vis, None
    
    elif im_type == "undistorded_images" :

        fish_params, _, _, _ = load_fisheye_params(gt_scene_path, frame)
        K_und, und_hw, _, _ = load_undistorted_pinhole_K(gt_scene_path, frame)
        
        depth_und = undistort_depth_to_pinhole(depth_dist, K_und, und_hw, fish_params)

        depth_metric = downsample_depth_preserving(depth_und, out_h, out_w)
        depth_vis = process_depth(depth_metric)

        K_518 = scale_K_pinhole(K_und, und_hw, (out_h, out_w))

        return depth_metric, depth_vis, K_518

    else :
        raise ValueError(f"im_type not recognized")

def get_image(frame_stem: str, scene_path: Path, im_type: str= "undistorded_images", out_h: int=518, out_w: int=518) -> np.ndarray:
    img_path = scene_path / im_type / "images" / f"{frame_stem}.JPG"
    img = Image.open(img_path).convert("RGB")
    img_ = img.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.array(img_)

def get_depth_pre(frame, pred_scene_path) :

    path = pred_scene_path / "depths" / f"{frame}.npy"
    pred_depth = np.load(path)

    depth = pred_depth.copy()
    depth[~np.isfinite(depth)] = np.nan
    valid_pixels = depth[np.isfinite(depth) & (depth > 0)]
    vmin, vmax = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
    depth_vis = np.clip((depth - vmin) / (vmax - vmin), 0, 1)

    return pred_depth, depth_vis

def get_mask(frame, gt_scene_path, target_size=518) :
    mask_path = gt_scene_path / "masks_for_images" / f"{frame}.png"
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
        
    return np.array(square_mask, dtype=np.uint8)

if __name__ == "__main__":

    dataset_path = Path("data/eth3D/")
    preds_path = Path("outputs/eth3D_local/")

    scenes = [s.name for s in preds_path.iterdir() if s.is_dir()]
    selected_scenes = random.sample(scenes, 5)

    imgs = []
    depths = []

    fig, axes = plt.subplots(3, 5, figsize=(10, 6))

    for i, scene in enumerate(selected_scenes):

        gt_scene_path = dataset_path / scene
        pred_scene_path = preds_path / scene

        pred_depth_path = preds_path / scene / "depths"
        frames = [p.stem for p in pred_depth_path.glob("*.npy")]
        frame = random.choice(frames)

        _, depth_visu_un, _= get_depth_gt(frame, gt_scene_path, im_type="undistorded_images")
        img_gt_un = get_image(frame, gt_scene_path, im_type="undistorded_images")

        mask_image = get_mask(frame, gt_scene_path)
        _, pred_depth = get_depth_pre(frame, pred_scene_path)

        axes[0, i].imshow(img_gt_un)
        axes[0, i].set_title(scene.upper(), fontsize=8, pad=2)
        axes[0, i].axis("off")

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color='white')
        im = axes[1, i].imshow(depth_visu_un, cmap=cmap, vmin=0, vmax=1)
        axes[1, i].axis("off")
        
        im = axes[2, i].imshow(pred_depth, cmap=cmap, vmin=0, vmax=1)
        axes[2, i].axis("off")

    plt.subplots_adjust(left=0.012, right=1, top=0.98, bottom=0, wspace=0.02, hspace=0.02)

    fig.text(0.002, 0.83, "RGB IMAGE", va="center", rotation=90,
         fontsize=8, fontweight="bold")
    fig.text(0.002, 0.50, "GROUND TRUTH DEPTH", va="center", rotation=90,
            fontsize=8, fontweight="bold")
    fig.text(0.002, 0.17, "VGGT PREDICTION", va="center", rotation=90,
            fontsize=8, fontweight="bold")
    plt.show()
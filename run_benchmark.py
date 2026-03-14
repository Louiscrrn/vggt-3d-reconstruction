from pathlib import Path
import numpy as np
from scripts.distortion import get_pm_depths
import open3d as o3d
import pandas as pd
import trimesh
import argparse


def load_confidence(frame: str, pred_scene_path: Path) -> np.ndarray:
    conf_path = pred_scene_path / "confidence" / f"{frame}.npy"
    return np.squeeze(np.load(conf_path).astype(np.float32))

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Umeyama (1991) : estime la Sim3 qui minimise ||dst - s*R@src + t||²
    src[k] ↔ dst[k]  (correspondance pixel-wise)

    Returns: R (3,3), t (3,), s (float), T (4,4)
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

    n      = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    cov     = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (np.sum(D * np.diag(S)) / var_src) if with_scale else 1.0
    t = mu_dst - s * R @ mu_src

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3,  3] = t

    return R, t, s, T

def align_pointmap(pred_pm, gt_pm, gt_depth, conf,
                   conf_threshold=0.5, with_scale=True):
    """
    Aligne pred_pm (espace VGGT) sur gt_pm (monde COLMAP) via Umeyama.
    Masque : gt_depth valide  &  confiance >= seuil

    Returns: pred_pm_aligned (H,W,3), T (4,4)
    """
    pred_pm  = np.squeeze(pred_pm).astype(np.float32)
    gt_pm    = np.squeeze(gt_pm).astype(np.float32)
    gt_depth = np.squeeze(gt_depth).astype(np.float32)
    conf     = np.squeeze(conf).astype(np.float32)

    mask_gt   = np.isfinite(gt_depth) & (gt_depth > 0)
    mask_pred = ((conf >= conf_threshold)
                 & np.isfinite(pred_pm[..., 0])
                 & np.isfinite(pred_pm[..., 1])
                 & np.isfinite(pred_pm[..., 2]))
    valid     = mask_gt & mask_pred

    n_valid = valid.sum()
    #print(f"  Pixels valides Umeyama : {n_valid}/{valid.size} "
    #      f"(GT={mask_gt.sum()}, conf≥{conf_threshold}={mask_pred.sum()})")
    if n_valid < 10:
        raise RuntimeError(f"Pas assez de correspondances : {n_valid}")

    src_pts = pred_pm[valid]
    dst_pts = gt_pm[valid]

    _, _, s, T = umeyama_alignment(src_pts, dst_pts, with_scale=with_scale)
    #print(f"  Scale estimé : {s:.4f}")

    H, W, _ = pred_pm.shape
    pts_h   = np.hstack([pred_pm.reshape(-1, 3),
                         np.ones((H * W, 1), dtype=np.float32)])
    aligned = (T @ pts_h.T).T[:, :3]

    return aligned.reshape(H, W, 3), T

def make_pcd(pm, mask, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pm[mask])
        pcd.paint_uniform_color(color)
        return pcd

def process_pm(gt_pm, pred_pm, gt_depth, conf, conf_threshold=0.5) :

    gt_pm    = np.squeeze(gt_pm).astype(np.float32)
    pred_pm  = np.squeeze(pred_pm).astype(np.float32)

    mask_gt   = np.isfinite(gt_depth) & (gt_depth > 0)
    mask_pred = ((conf >= conf_threshold)
                 & np.isfinite(pred_pm[..., 0])
                 & np.isfinite(pred_pm[..., 1])
                 & np.isfinite(pred_pm[..., 2]))

    pcd_gt = make_pcd(gt_pm,  mask_gt,   [0.0, 0.8, 0.0])
    pcd_pm = make_pcd(pred_pm, mask_pred, [0.8, 0.1, 0.1])

    return pcd_gt, pcd_pm

def show_pointmaps_comparison_o3d(gt_pm, pred_pm, gt_depth, conf,
                                   conf_threshold=0.5):
    
    pcd_gt, pcd_pm = process_pm(gt_pm, pred_pm, gt_depth, conf, conf_threshold)

    o3d.visualization.draw_geometries(
        [pcd_gt,pcd_pm],
        window_name="GT (vert) vs Pred alignée (rouge)",
        width=1280, height=720,
    )

def compute_metrics(pm_gt, pm_pred, gt_depth, conf, conf_threshold, frame, scene) :

    pcd_gt, pcd_pred = process_pm(pm_gt, pm_pred, gt_depth, conf, conf_threshold)

    dists_acc = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    dists_comp = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))

    acc = float(dists_acc.mean()) if len(dists_acc) > 0 else np.nan
    comp = float(dists_comp.mean()) if len(dists_comp) > 0 else np.nan
    overall = float(0.5 * (acc + comp)) if np.isfinite(acc) and np.isfinite(comp) else np.nan

    return {"Acc" : acc, "Comp" : comp, "Overall" : overall, "Frame" : frame, "Scene" : scene}

def extract_points_from_pm(gt_pm, pred_pm, gt_depth, conf, conf_threshold=0.5):
    """
    Retourne deux arrays (N,3):
    - gt_pts
    - pred_pts (déjà alignés si pred_pm est alignée)
    """
    gt_pm    = np.squeeze(gt_pm).astype(np.float32)
    pred_pm  = np.squeeze(pred_pm).astype(np.float32)
    gt_depth = np.squeeze(gt_depth).astype(np.float32)
    conf     = np.squeeze(conf).astype(np.float32)

    mask_gt = (
        np.isfinite(gt_depth) & (gt_depth > 0) &
        np.isfinite(gt_pm[..., 0]) & np.isfinite(gt_pm[..., 1]) & np.isfinite(gt_pm[..., 2])
    )
    mask_pred = (
        (conf >= conf_threshold) &
        np.isfinite(pred_pm[..., 0]) & np.isfinite(pred_pm[..., 1]) & np.isfinite(pred_pm[..., 2])
    )

    gt_pts = gt_pm[mask_gt]
    pred_pts = pred_pm[mask_pred]
    return gt_pts, pred_pts

def save_pointcloud_ply_trimesh(points: np.ndarray, out_path: Path, color_rgb=None):
    """
    Sauvegarde un nuage de points en .ply via trimesh.
    color_rgb: [R,G,B] optionnel (0-255), appliqué à tous les points
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points doit être de forme (N,3), reçu {points.shape}")

    if len(points) == 0:
        raise ValueError("Nuage vide, rien à sauvegarder.")

    kwargs = {}
    if color_rgb is not None:
        color = np.array(color_rgb, dtype=np.uint8).reshape(1, 3)
        colors = np.repeat(color, len(points), axis=0)
        kwargs["colors"] = colors

    pc = trimesh.points.PointCloud(vertices=points, **kwargs)
    pc.export(out_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmarking VGGT results")
    parser.add_argument("--dataset_path", type=str, default="data/eth3D/", help="Path to ground truth data")
    parser.add_argument("--preds_path", type=str, default="outputs/eth3D_local/", help="Path to inference predictions")
    parser.add_argument("--conf_threshold", type=float, default=1.0, help="Confidence threshold for evaluation")
    
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    preds_path   = Path(args.preds_path)
    conf_threshold = args.conf_threshold
    
    data = []
    for scene_pred_dir in sorted(preds_path.iterdir()):
        if not scene_pred_dir.is_dir():
            continue

        scene_name      = scene_pred_dir.name
        gt_scene_path   = dataset_path / scene_name
        pred_scene_path = preds_path   / scene_name
        pm_dir          = scene_pred_dir / "pointmaps"

        print(f"# Scene: {scene_name}")    

        all_gt_points = []
        all_pred_points = []
        scene_data = []
        frames = [f.stem for f in sorted(pm_dir.glob("*.npy"))]
        for frame in frames:

            gt_pm, gt_depth, pred_pm, pred_depth = get_pm_depths(
                frame, gt_scene_path, pred_scene_path
            )
            conf = load_confidence(frame, pred_scene_path)

            # Alignement Umeyama
            pred_pm_aligned, T = align_pointmap(
                pred_pm, gt_pm, gt_depth, conf,
                conf_threshold=conf_threshold,
                with_scale=True,
            )

            # Compute the metrics
            hist = compute_metrics(gt_pm, pred_pm_aligned, gt_depth, conf, conf_threshold, frame, scene_name)
            scene_data.append(hist)
            data.append(hist)

            gt_pts_frame, pred_pts_frame = extract_points_from_pm(
                gt_pm, pred_pm_aligned, gt_depth, conf, conf_threshold
            )
            all_gt_points.append(gt_pts_frame)
            all_pred_points.append(pred_pts_frame)

            # Visualisation
            """show_pointmaps_comparison_o3d(
                gt_pm, pred_pm_aligned, gt_depth, conf,
                conf_threshold=conf_threshold,
            )
            """
        df_scene = pd.DataFrame(scene_data)
        s_acc = df_scene["Acc"].mean()
        s_comp = df_scene["Comp"].mean()
        s_ov = df_scene["Overall"].mean()
        print(f"Scene {scene_name} - Acc.↓ {s_acc:.4f} | Comp.↓ {s_comp:.4f} | Overall↓ {s_ov:.4f}")

        # Concatenate and export
        gt_scene_pts = np.concatenate(all_gt_points, axis=0)
        pred_scene_pts = np.concatenate(all_pred_points, axis=0)

        save_pointcloud_ply_trimesh(
                gt_scene_pts,
                pred_scene_path / f"{scene_name}_gt.ply",
                color_rgb=[0, 255, 0],   
            )
        save_pointcloud_ply_trimesh(
                pred_scene_pts,
                pred_scene_path / f"{scene_name}_pred.ply",
                color_rgb=[255, 0, 0],  
            )
    
    df_final = pd.DataFrame(data)
    total_acc = df_final["Acc"].mean()
    total_comp = df_final["Comp"].mean()
    total_ov = df_final["Overall"].mean()

    print(f"\n" + "="*80)
    print(f"GLOBAL SCORE (All scenes combined)")
    print(f"Acc.↓ {total_acc:.4f} | Comp.↓ {total_comp:.4f} | Overall↓ {total_ov:.4f}")
    print("="*80)
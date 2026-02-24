from pathlib import Path
import numpy as np
from distortion_utils import get_maps_depths, show_pointmap_o3d
import open3d as o3d


# ─────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────

def load_confidence(frame: str, pred_scene_path: Path) -> np.ndarray:
    conf_path = pred_scene_path / "confidence" / f"{frame}.npy"
    return np.squeeze(np.load(conf_path).astype(np.float32))


# ─────────────────────────────────────────────────────────────
# Umeyama
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Alignement
# ─────────────────────────────────────────────────────────────

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
    print(f"  Pixels valides Umeyama : {n_valid}/{valid.size} "
          f"(GT={mask_gt.sum()}, conf≥{conf_threshold}={mask_pred.sum()})")
    if n_valid < 10:
        raise RuntimeError(f"Pas assez de correspondances : {n_valid}")

    src_pts = pred_pm[valid]
    dst_pts = gt_pm[valid]

    _, _, s, T = umeyama_alignment(src_pts, dst_pts, with_scale=with_scale)
    print(f"  Scale estimé : {s:.4f}")

    H, W, _ = pred_pm.shape
    pts_h   = np.hstack([pred_pm.reshape(-1, 3),
                         np.ones((H * W, 1), dtype=np.float32)])
    aligned = (T @ pts_h.T).T[:, :3]

    return aligned.reshape(H, W, 3), T


# ─────────────────────────────────────────────────────────────
# Métriques  (pixel-wise + NN-based)
# ─────────────────────────────────────────────────────────────

def compute_metrics(pred_pm_aligned: np.ndarray,
                    gt_pm: np.ndarray,
                    gt_depth: np.ndarray,
                    conf: np.ndarray,
                    conf_threshold: float = 0.5,
                    thresholds: tuple = (0.05, 0.10, 0.20)):
    """
    Calcule accuracy et complétude après alignement.

    • Accuracy  : pour chaque point PRED valide, distance au point GT
                  correspondant (pixel-wise). Fraction en dessous du seuil τ.
    • Complétude: pour chaque point GT valide, distance au point PRED aligné
                  le plus proche (NN dans o3d). Fraction en dessous de τ.
    • Mean / Median / RMSE sur les distances pixel-wise (pixels valides des deux).

    Args:
        pred_pm_aligned : (H,W,3)  pointmap prédit aligné (monde)
        gt_pm           : (...,H,W,3)
        gt_depth        : (...,H,W)
        conf            : (H,W)
        conf_threshold  : seuil confiance
        thresholds      : seuils (en mètres) pour acc/completude

    Returns: dict de métriques
    """
    pred_pm_aligned = np.squeeze(pred_pm_aligned).astype(np.float32)
    gt_pm           = np.squeeze(gt_pm).astype(np.float32)
    gt_depth        = np.squeeze(gt_depth).astype(np.float32)
    conf            = np.squeeze(conf).astype(np.float32)

    mask_gt   = np.isfinite(gt_depth) & (gt_depth > 0)
    mask_pred = ((conf >= conf_threshold)
                 & np.isfinite(pred_pm_aligned[..., 0])
                 & np.isfinite(pred_pm_aligned[..., 1])
                 & np.isfinite(pred_pm_aligned[..., 2]))

    # ── Accuracy pixel-wise (pixels valides dans les deux) ──────────────
    valid_both = mask_gt & mask_pred
    diff       = pred_pm_aligned[valid_both] - gt_pm[valid_both]   # (N,3)
    dist_pw    = np.linalg.norm(diff, axis=1)                      # (N,)

    metrics = {
        "n_valid_both" : int(valid_both.sum()),
        "n_gt_valid"   : int(mask_gt.sum()),
        "n_pred_valid" : int(mask_pred.sum()),
        "mean_err"     : float(dist_pw.mean()),
        "median_err"   : float(np.median(dist_pw)),
        "rmse"         : float(np.sqrt(np.mean(dist_pw ** 2))),
    }

    for τ in thresholds:
        metrics[f"acc@{τ:.2f}m"] = float((dist_pw < τ).mean() * 100)

    # ── Complétude NN (o3d KD-tree) ──────────────────────────────────────
    #   Pour chaque point GT, distance au pred aligné le plus proche
    pred_pts = pred_pm_aligned[mask_pred]   # (M,3)
    gt_pts   = gt_pm[mask_gt]               # (K,3)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts)
    tree = o3d.geometry.KDTreeFlann(pcd_pred)

    dist_comp = np.empty(len(gt_pts), dtype=np.float32)
    for i, pt in enumerate(gt_pts):
        _, _, d2 = tree.search_knn_vector_3d(pt, 1)
        dist_comp[i] = np.sqrt(d2[0])

    for τ in thresholds:
        metrics[f"comp@{τ:.2f}m"] = float((dist_comp < τ).mean() * 100)

    metrics["mean_comp_err"]   = float(dist_comp.mean())
    metrics["median_comp_err"] = float(np.median(dist_comp))

    return metrics


def print_metrics_table(all_metrics: dict,
                        thresholds: tuple = (0.05, 0.10, 0.20)):
    """
    Affiche un tableau récap par frame puis les moyennes globales.

    all_metrics : { frame_name : metrics_dict }
    """
    τ_labels_acc  = [f"acc@{τ:.2f}m"  for τ in thresholds]
    τ_labels_comp = [f"comp@{τ:.2f}m" for τ in thresholds]

    col_keys = (["mean_err", "median_err", "rmse"]
                + τ_labels_acc
                + ["mean_comp_err", "median_comp_err"]
                + τ_labels_comp)

    col_heads = (["mean↓", "med↓", "rmse↓"]
                 + [f"acc%@{τ:.2f}↑" for τ in thresholds]
                 + ["c.mean↓", "c.med↓"]
                 + [f"cmp%@{τ:.2f}↑" for τ in thresholds])

    # Largeurs colonnes
    frame_w = max(len(k) for k in all_metrics) + 2
    col_w   = [max(len(h), 9) + 1 for h in col_heads]

    sep  = "─" * (frame_w + sum(col_w) + len(col_w) + 1)
    fmt_h = f"{'Frame':<{frame_w}}" + "".join(f" {h:>{w}}" for h, w in zip(col_heads, col_w))
    fmt_f = (lambda name, m:
             f"{name:<{frame_w}}"
             + "".join(
                 f" {m.get(k, float('nan')):>{w}.3f}"
                 if "%" not in k else
                 f" {m.get(k, float('nan')):>{w}.1f}"
                 for k, w in zip(col_keys, col_w)
             ))

    print(f"\n{'═'*len(sep)}")
    print(f"{'MÉTRIQUES PAR FRAME':^{len(sep)}}")
    print(f"{'═'*len(sep)}")
    print(fmt_h)
    print(sep)

    for frame_name, m in all_metrics.items():
        print(fmt_f(frame_name, m))

    # ── Moyennes globales ──
    if len(all_metrics) > 1:
        avg = {}
        for k in col_keys:
            vals = [m[k] for m in all_metrics.values() if k in m]
            avg[k] = float(np.mean(vals)) if vals else float("nan")
        print(sep)
        print(fmt_f("MOYENNE", avg))

    print(f"{'═'*len(sep)}\n")


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def show_pointmaps_comparison_o3d(gt_pm, pred_pm, gt_depth, conf,
                                   conf_threshold=0.5):
    gt_pm    = np.squeeze(gt_pm).astype(np.float32)
    pred_pm  = np.squeeze(pred_pm).astype(np.float32)
    gt_depth = np.squeeze(gt_depth).astype(np.float32)
    conf     = np.squeeze(conf).astype(np.float32)

    mask_gt   = np.isfinite(gt_depth) & (gt_depth > 0)
    mask_pred = ((conf >= conf_threshold)
                 & np.isfinite(pred_pm[..., 0])
                 & np.isfinite(pred_pm[..., 1])
                 & np.isfinite(pred_pm[..., 2]))

    def make_pcd(pm, mask, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pm[mask])
        pcd.paint_uniform_color(color)
        return pcd

    o3d.visualization.draw_geometries(
        [make_pcd(gt_pm,  mask_gt,   [0.0, 0.8, 0.0]),
         make_pcd(pred_pm, mask_pred, [0.8, 0.1, 0.1])],
        window_name="GT (vert) vs Pred alignée (rouge)",
        width=1280, height=720,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.5
THRESHOLDS     = (0.05, 0.10, 0.20)   # mètres

if __name__ == "__main__":

    dataset_path = Path("data/eth3D/")
    preds_path   = Path("outputs/eth3D_local/")

    for scene_pred_dir in sorted(preds_path.iterdir()):
        if not scene_pred_dir.is_dir():
            continue

        scene_name      = "office"
        scene_pred_dir  = preds_path / scene_name
        scene_name      = scene_pred_dir.name
        gt_scene_path   = dataset_path / scene_name
        pred_scene_path = preds_path   / scene_name
        pm_dir          = scene_pred_dir / "pointmaps"

        print(f"\n{'#'*60}")
        print(f"# Scene: {scene_name}")
        print(f"{'#'*60}")

        all_frame_metrics = {}

        frames = [f.stem for f in sorted(pm_dir.glob("*.npy"))]
        for frame in frames:

            print(f"\n  ── Frame: {frame} ──")

            gt_pm, gt_depth, pred_pm, pred_depth = get_maps_depths(
                frame, gt_scene_path, pred_scene_path
            )
            conf = load_confidence(frame, pred_scene_path)
            print(f"  Confiance — min:{conf.min():.3f}  max:{conf.max():.3f}  mean:{conf.mean():.3f}")

            # Alignement Umeyama
            pred_pm_aligned, T = align_pointmap(
                pred_pm, gt_pm, gt_depth, conf,
                conf_threshold=CONF_THRESHOLD,
                with_scale=True,
            )

            # Métriques
            metrics = compute_metrics(
                pred_pm_aligned, gt_pm, gt_depth, conf,
                conf_threshold=CONF_THRESHOLD,
                thresholds=THRESHOLDS,
            )
            all_frame_metrics[frame] = metrics

            # Visualisation
            #show_pointmap_o3d(gt_pm,   gt_depth)
            #show_pointmap_o3d(pred_pm, pred_depth)
            show_pointmaps_comparison_o3d(
                gt_pm, pred_pm_aligned, gt_depth, conf,
                conf_threshold=CONF_THRESHOLD,
            )


        # Tableau récap de la scène
        print_metrics_table(all_frame_metrics, thresholds=THRESHOLDS)

        break
from pathlib import Path
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from utils import load_eth3d_pose_logic


# ==========================================================
# Umeyama Sim(3)
# ==========================================================
def umeyama(X, Y):
    X = X.T
    Y = Y.T

    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)

    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = np.sum(Xc ** 2) / X.shape[1]
    cov_xy = (Yc @ Xc.T) / X.shape[1]

    U, D, Vt = np.linalg.svd(cov_xy)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - s * R @ mu_x

    return s, R, t.flatten()


# ==========================================================
# Projection 3D → 2D
# ==========================================================
def project_points(points_world, R, t, K):
    points_cam = (R @ points_world.T + t.reshape(3,1)).T

    mask_front = points_cam[:,2] > 0
    points_cam = points_cam[mask_front]

    proj = (K @ points_cam.T).T
    proj[:,0] /= proj[:,2]
    proj[:,1] /= proj[:,2]

    return proj[:,:2], mask_front


# ==========================================================
# Construire GT visible
# ==========================================================
def build_visible_gt(scan_points, frames, scene_path, load_pose_fn):

    visible_points = []

    for frame in frames:

        R_gt, t_gt, K_gt, _, _ = load_pose_fn(frame, scene_path)

        mask_path = scene_path / "masks_for_images" / f"{frame}.png"
        mask_img = np.array(Image.open(mask_path))
        H, W = mask_img.shape

        uv, mask_front = project_points(scan_points, R_gt, t_gt, K_gt)

        uv = uv.astype(int)
        points_front = scan_points[mask_front]

        # Dans image
        valid = (
            (uv[:,0] >= 0) &
            (uv[:,0] < W) &
            (uv[:,1] >= 0) &
            (uv[:,1] < H)
        )

        uv = uv[valid]
        points_valid = points_front[valid]

        # Masque ETH3D
        mask_vals = mask_img[uv[:,1], uv[:,0]]
        mask_ok = mask_vals == 0

        visible_points.append(points_valid[mask_ok])

    if len(visible_points) == 0:
        return np.empty((0,3))

    return np.concatenate(visible_points, axis=0)


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    dataset_path = Path("data/eth3D/")
    preds_path = Path("outputs/eth3D_local/")

    for scene_dir in preds_path.iterdir():
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        print(f"\n--- Evaluating Scene: {scene_name} ---")

        pred_points = np.asarray(trimesh.load(scene_dir / "scan.ply").vertices)

        gt_scene_path = dataset_path / scene_name
        gt_points = np.asarray(trimesh.load(gt_scene_path / "scan.ply").vertices)

        # Frames utilisées par VGGT
        depth_folder = scene_dir / "depths"
        frames = [f.stem for f in depth_folder.glob("*.jpg")]

        print("Frames used:", len(frames))

        # Construire GT visible
        gt_visible = build_visible_gt(
            gt_points,
            frames,
            gt_scene_path,
            load_eth3d_pose_logic  # ta fonction existante
        )

        print("GT visible points:", len(gt_visible))

        if len(gt_visible) == 0:
            print("No visible GT points.")
            continue

        # ==========================================================
        # Alignement Umeyama + NN
        # ==========================================================
        pred_current = pred_points.copy()

        total_scale = 1.0
        total_R = np.eye(3)
        total_t = np.zeros(3)

        iterations = 5

        for it in range(iterations):
            print(f"\nIteration {it+1}")

            nn = NearestNeighbors(n_neighbors=1).fit(gt_visible)
            distances, indices = nn.kneighbors(pred_current)
            gt_corr = gt_visible[indices[:,0]]

            # Filtrage robuste
            threshold = np.percentile(distances, 85)
            mask = distances.flatten() < threshold

            pred_valid = pred_current[mask]
            gt_valid = gt_corr[mask]

            print("Used correspondences:", len(pred_valid))

            s, R, t = umeyama(pred_valid, gt_valid)

            total_scale *= s
            total_R = R @ total_R
            total_t = s * (R @ total_t) + t

            pred_current = s * (pred_current @ R.T) + t

        pred_aligned = total_scale * (pred_points @ total_R.T) + total_t

        print("Final scale:", total_scale)

        # Sauvegarde
        trimesh.PointCloud(pred_aligned).export("pred_aligned.ply")
        trimesh.PointCloud(gt_visible).export("gt_visible.ply")

        print("Saved aligned clouds.")
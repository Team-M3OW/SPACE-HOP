import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from dataloader import SpeedDataset
from model import FastPoseViT
from geometry_utils import (
    recover_true_rotation,
    recover_translation_components,
    axis_angle_to_matrix
)
from hopf_grid import get_closest_anchor


# -----------------------------
# --- Utility Functions
# -----------------------------

def rotmat2quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = r00 + r11 + r22
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return q


def translation_error_euclidian(T_pred, T_gt):
    return torch.norm(T_pred - T_gt, dim=1)


def rotation_error_geodesic(R_pred, R_gt):
    q_pred = rotmat2quat(R_pred)
    q_gt = rotmat2quat(R_gt)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    E_r = 2 * torch.acos(dot)
    return torch.rad2deg(E_r)


# -----------------------------
# --- Validation Function
# -----------------------------

@torch.no_grad()
def validate(model, loader, device, intrinsics):

    model.eval()

    total_t_error = []
    total_r_error = []

    fx, fy, cx, cy = intrinsics
    K_batch = torch.eye(3, device=device).unsqueeze(0)
    K_batch[0, 0, 0] = float(fx)
    K_batch[0, 1, 1] = float(fy)
    K_batch[0, 0, 2] = float(cx)
    K_batch[0, 1, 2] = float(cy)

    for batch in tqdm(loader, desc="Validation"):

        images = batch["pixel_values"].to(device)
        R_apparent_gt = batch["R_apparent_gt"].to(device)
        T_gt = batch["T_gt"].to(device)
        R_gt_true = batch["R_gt_true"].to(device)
        crop_params = batch["crop_params"].to(device)

        outputs = model(images)

        U_pred = outputs["U"]
        logits = outputs["logits"]
        deltas = outputs["deltas"]
        anchors = outputs["anchors"]

        # -------- Argmax inference --------
        winning_bins = torch.argmax(logits, dim=-1)
        batch_indices = torch.arange(images.shape[0], device=device)

        R_base = anchors[winning_bins]
        delta_win = deltas[batch_indices, winning_bins, :]
        R_delta = axis_angle_to_matrix(delta_win)
        R_apparent_pred = torch.bmm(R_base, R_delta)

        # -------- Recover true pose --------
        T_pred = recover_translation_components(
            U_pred, crop_params, (1920, 1200), (224, 224), K_batch
        )

        R_pred_true = recover_true_rotation(R_apparent_pred, T_pred)

        # -------- Errors --------
        t_err = translation_error_euclidian(T_pred, T_gt)
        r_err = rotation_error_geodesic(R_pred_true, R_gt_true)

        total_t_error.append(t_err.cpu())
        total_r_error.append(r_err.cpu())

    total_t_error = torch.cat(total_t_error)
    total_r_error = torch.cat(total_r_error)

    print("\n===== FINAL METRICS =====")
    print(f"Translation Error (mean): {total_t_error.mean():.4f} m")
    print(f"Rotation Error (mean):    {total_r_error.mean():.4f} deg")
    print(f"Median T Error:           {total_t_error.median():.4f} m")
    print(f"Median R Error:           {total_r_error.median():.4f} deg")

    return total_t_error.mean().item(), total_r_error.mean().item()


# -----------------------------
# --- Main
# -----------------------------

if __name__ == "__main__":

    config = {
        "val_csv": "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/synthetic/validation.csv",
        "img_folder": "~/speed/",
        "intrinsic_file": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/camera.json",
        "keypoints_file": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/tangoPoints.mat",
        "checkpoint": "./checkpoints_SEPIA/best_model.pth",
        "jepa_weights": "/mnt/external_ssd/ARSH_ARNABI/jepa_joints_speedplus-ep300.pth.tar",
        "batch_size": 8,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeedDataset(
        csv_file=config["val_csv"],
        image_root=config["img_folder"],
        intrinsics_file=config["intrinsic_file"],
        keypoints_file=config["keypoints_file"],
        mode="val"
    )

    loader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    # --- Intrinsics ---
    if hasattr(dataset, 'intrinsics_mat'):
        K = dataset.intrinsics_mat
        intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    else:
        K = dataset.intrinsics
        if torch.is_tensor(K):
            K = K.numpy()
        intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])

    # --- Model ---
    model = FastPoseViT(
        jepa_path=config["jepa_weights"],
        img_size=224,
        patch_size=16
    )

    checkpoint = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    validate(model, loader, device, intrinsics)
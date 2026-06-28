import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# Import your existing, untouched scripts
from dataloader import SpeedDataset
from model import FastPoseViT
from geometry_utils import (
    recover_true_rotation, 
    recover_translation_components, 
    axis_angle_to_matrix
)

# =========================================================================
# SELF-CONTAINED MATH UTILS
# =========================================================================
def rotmat2quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert R.shape[-2:] == (3, 3)
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = r00 + r11 + r22
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    mask = (trace <= 0) & (r00 >= r11) & (r00 >= r22)
    s = torch.sqrt(1.0 + r00[mask] - r11[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 1] = 0.25 * s
    q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s

    mask = (trace <= 0) & (r11 > r00) & (r11 >= r22)
    s = torch.sqrt(1.0 + r11[mask] - r00[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 2] = 0.25 * s
    q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s

    mask = (trace <= 0) & (r22 > r00) & (r22 > r11)
    s = torch.sqrt(1.0 + r22[mask] - r00[mask] - r11[mask]) * 2.0
    q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    q[mask, 3] = 0.25 * s
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

# =========================================================================
# PLOTTING LOGIC
# =========================================================================
def draw_error_histograms(error_dict, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), facecolor='white')
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    datasets = list(error_dict.keys())
    colors = ['#4C72B0', '#55A868', '#C44E52'] 
    
    for row_idx, ds_name in enumerate(datasets):
        t_errs = error_dict[ds_name]['t']
        r_errs = error_dict[ds_name]['r']
        
        # Translation Hist
        ax_t = axes[row_idx, 0]
        ax_t.hist(t_errs, bins=50, color=colors[row_idx], edgecolor='black', alpha=0.7)
        ax_t.set_title(f"{ds_name} - Translation Error", fontweight='bold')
        ax_t.set_xlabel("Error (meters)")
        ax_t.set_ylabel("Number of Images")
        ax_t.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotation Hist
        ax_r = axes[row_idx, 1]
        ax_r.hist(r_errs, bins=50, color=colors[row_idx], edgecolor='black', alpha=0.7)
        ax_r.set_title(f"{ds_name} - Rotation Error", fontweight='bold')
        ax_r.set_xlabel("Error (degrees)")
        ax_r.set_ylabel("Number of Images")
        ax_r.grid(axis='y', linestyle='--', alpha=0.7)
        ax_r.set_yscale('log') # Log scale to see the symmetry outliers

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nDONE! Saved Histograms to: {save_path}")

# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    config = {
        "val_csv" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/synthetic/validation.csv",
        "sunlamp_csv" :"/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/sunlamp/sunlamp.csv",
        "lightbox_csv" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/lightbox/lightbox.csv",
        "img_folder" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/",
        "intrinsic_file" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/camera.json",
        "keypoints_file": "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/tangoPoints.mat",
        "checkpoint": "/home/aac/shared/teams/dtu/mlr-lab/SEPIA/checkpoints_SEPIA/checkpoint_epoch_300.pth", 
        "batch_size" : 32 # Bumped up for faster purely-numeric inference
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastPoseViT(img_size=224, patch_size=16)
    
    if not os.path.exists(config["checkpoint"]):
        print(f"Error: Could not find checkpoint at {config['checkpoint']}")
        return

    checkpoint = torch.load(config["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model Loaded Successfully for Distribution Analysis.")

    datasets = {
        "SYNTHETIC": config["val_csv"],
        "SUNLAMP": config["sunlamp_csv"],
        "LIGHTBOX": config["lightbox_csv"]
    }
    
    error_distributions = {ds: {'t': [], 'r': []} for ds in datasets.keys()}
    
    K_tuple = (2988.5795, 2988.3401, 960.0, 600.0)
    fx, fy, cx, cy = K_tuple
    K_batch = torch.eye(3, device=device).unsqueeze(0)
    K_batch[0, 0, 0] = float(fx); K_batch[0, 1, 1] = float(fy)
    K_batch[0, 0, 2] = float(cx); K_batch[0, 1, 2] = float(cy)

    for ds_name, csv_path in datasets.items():
        print(f"\nExtracting Errors from {ds_name}...")
        ds = SpeedDataset(
            csv_file=csv_path, image_root=config["img_folder"], 
            intrinsics_file=config["intrinsic_file"], keypoints_file=config["keypoints_file"], mode='val'
        )
        loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{ds_name} Inference", leave=False):
                images = batch['pixel_values'].to(device)
                crop_params = batch['crop_params'].to(device)
                T_gt = batch['T_gt'].to(device)
                R_gt_true = batch['R_gt_true'].to(device)
                
                outputs = model(images)
                u_pred = outputs['U']
                logits = outputs['logits']
                pred_deltas = outputs['deltas']
                anchors = outputs['anchors']
                
                winning_bins = torch.argmax(logits, dim=-1)
                batch_indices = torch.arange(images.shape[0], device=device)
                
                R_base = anchors[winning_bins]
                delta_win = pred_deltas[batch_indices, winning_bins, :]
                R_delta = axis_angle_to_matrix(delta_win)
                R_apparent_pred = torch.bmm(R_base, R_delta)
                
                T_pred = recover_translation_components(
                    u_pred, crop_params, (1920, 1200), (224, 224), K_batch
                )
                R_pred_true = recover_true_rotation(R_apparent_pred, T_pred)
                
                t_errs = translation_error_euclidian(T_pred, T_gt)
                r_errs = rotation_error_geodesic(R_pred_true, R_gt_true)
                
                # Fast append without storing images
                error_distributions[ds_name]['t'].extend(t_errs.cpu().tolist())
                error_distributions[ds_name]['r'].extend(r_errs.cpu().tolist())

    draw_error_histograms(error_distributions, "results/SEPIA_Error_Distributions.png")

if __name__ == "__main__":
    main()
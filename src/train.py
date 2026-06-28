import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils 
from tqdm import tqdm
import numpy as np
import os
import wandb
from dataloader import SpeedDataset
from model import FastPoseViT
from box import draw_satellite_wireframe
from hopf_grid import get_closest_anchor
from geometry_utils import (
    recover_true_rotation, 
    recover_translation_components, 
    ortho6d_to_matrix,
    get_gt_deltas, 
    axis_angle_to_matrix
) 

def translation_loss(u_pred, u_gt):
    return torch.sum((u_pred - u_gt) ** 2, dim=1).mean()

def rotation_loss(r6d_pred, r6d_gt):
    return torch.sum((r6d_pred - r6d_gt) ** 2, dim=1).mean()

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
    return torch.norm(T_pred - T_gt, dim=1).mean()

def rotation_error_geodesic(R_pred, R_gt):
    q_pred = rotmat2quat(R_pred)
    q_gt = rotmat2quat(R_gt)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    E_r = 2 * torch.acos(dot)
    return torch.rad2deg(E_r).mean()

def save_debug_images(images, path="debug_input_batch.png"):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    denorm_imgs = images * std + mean
    denorm_imgs = torch.clamp(denorm_imgs, 0, 1)
    vutils.save_image(denorm_imgs, path, nrow=8)
    print(f"[DEBUG] Saved input batch grid to {path}")

def get_optimizer(model, config):
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        # Catch all three of our new decoupled heads
        # In train.py inside get_optimizer()
        is_head = any(h in name for h in ["pose_head", "coarse_cross_attn", "coarse_classifier", "fine_head"])        
        # MUON CONSTRAINT: Only supports 2D parameters. 
        # Biases, 1D vectors, and 4D Conv filters MUST go to AdamW.
        if (p.ndim != 2 or 
            "bias" in name or 
            is_head or 
            "pos_embed" in name or 
            "cls_token" in name or
            "norm" in name):
            adamw_params.append(p)
        else:
            muon_params.append(p)

    # Heads, biases, norms, and the 4D PatchEmbed conv go to AdamW
    opt1 = torch.optim.AdamW(adamw_params, lr=config["lr"], weight_decay=0.01)
    
    # Core 2D transformer weight matrices go to Muon
    opt2 = torch.optim.Muon(muon_params, lr=config["lr"], weight_decay=0.0)
    
    return opt1, opt2

def compute_hopf_losses(logits, pred_deltas, anchors, R_apparent_gt):
    B = logits.shape[0]
    
    # 1. Coarse Target
    target_bins = get_closest_anchor(R_apparent_gt, anchors)
    loss_coarse = F.cross_entropy(logits, target_bins)

    # Coarse Accuracy (Diagnostic)
    winning_bins = torch.argmax(logits, dim=-1)
    coarse_acc = (winning_bins == target_bins).float().mean()
    
    # 2. Fine Target (Only supervise the bin that matches the GT)
    batch_indices = torch.arange(B, device=logits.device)
    active_pred_deltas = pred_deltas[batch_indices, target_bins, :]
    target_anchors = anchors[target_bins]
    
    with torch.no_grad():
        # Relative rotation from anchor to true apparent
        R_rel_true = torch.bmm(target_anchors.transpose(1, 2), R_apparent_gt)
        true_deltas = get_gt_deltas(R_rel_true)
        
    loss_fine = F.mse_loss(active_pred_deltas, true_deltas)
    # Fine Magnitude (Diagnostic)
    fine_mag = torch.norm(active_pred_deltas, dim=-1).mean()
    return loss_coarse, loss_fine, coarse_acc, fine_mag

def validate(model, loader, device, intrinsics, epoch, name="val"):
    model.eval()
    val_loss = 0.0
    val_rot_loss = 0.0
    val_trans_loss = 0.0
    total_t_error = 0.0
    total_r_error = 0.0
    fx, fy, cx, cy = intrinsics
    K_batch = torch.eye(3, device=device).unsqueeze(0)
    K_batch[0, 0, 0] = float(fx)
    K_batch[0, 1, 1] = float(fy)
    K_batch[0, 0, 2] = float(cx)
    K_batch[0, 1, 2] = float(cy)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
            images = batch['pixel_values'].to(device)
            R_apparent_gt = batch['R_apparent_gt'].to(device)
            u_gt = batch['U_gt'].to(device)
            
            outputs = model(images)
            u_pred = outputs['U']
            logits = outputs['logits']
            pred_deltas = outputs['deltas']
            anchors = outputs['anchors']
            
            # --- Inference: Argmax -> Exp Map -> True Recovery ---
            winning_bins = torch.argmax(logits, dim=-1)
            batch_indices = torch.arange(images.shape[0], device=device)
            R_base = anchors[winning_bins]
            delta_win = pred_deltas[batch_indices, winning_bins, :]
            
            R_delta = axis_angle_to_matrix(delta_win)
            R_apparent_pred = torch.bmm(R_base, R_delta)
            
            l_trans = translation_loss(u_pred, u_gt)
            l_coarse, l_fine, coarse_acc, fine_mag = compute_hopf_losses(logits, pred_deltas, anchors, R_apparent_gt)
            l_rot = l_coarse + (10.0 * l_fine) # Weight the fine loss
            
            val_loss += (l_trans + l_rot).item()
            val_rot_loss += l_rot.item()
            val_trans_loss += l_trans.item()
            
            if 'T_gt' in batch and 'R_gt_true' in batch:
                T_gt = batch['T_gt'].to(device)
                R_gt_true = batch['R_gt_true'].to(device)
                crop_params = batch['crop_params'].to(device)
                
                T_pred = recover_translation_components(
                    u_pred, crop_params, (1920, 1200), (224, 224), K_batch
                )
                
                # RECOVER TRUE ROTATION USING FASTPOSE LOGIC
                R_pred_true = recover_true_rotation(R_apparent_pred, T_pred)
                
                t_err = translation_error_euclidian(T_pred, T_gt)
                total_t_error += t_err.item()
                
                r_err = rotation_error_geodesic(R_pred_true, R_gt_true)
                total_r_error += r_err.item()
                
                if (epoch % 10 == 0) and (batch_idx == 0): 
                    vis_images = []
                    num_vis = min(8, images.shape[0]) 
                    
                    for i in range(num_vis):
                        # Draws both Green (GT) and Red (Pred) wireframes on the same image
                        overlay_img = draw_satellite_wireframe(
                            images[i], 
                            R_gt_true[i], T_gt[i], 
                            R_pred_true[i], T_pred[i], 
                            K_batch[0], 
                            bbox=crop_params[i]
                        )
                        vis_images.append(wandb.Image(overlay_img, caption=f"Epoch {epoch} | Green: GT | Red: Pred"))
                    
                    wandb.log({f"{name}/pose_visualizations": vis_images, "epoch": epoch})
    avg_loss = val_loss / len(loader)
    
    if total_t_error > 0:
        avg_t_error = total_t_error / len(loader)
        avg_r_error = total_r_error / len(loader)
        print(f"Val Metrics | Loss: {avg_loss:.4f} | T_Err: {avg_t_error:.4f} m | R_Err: {avg_r_error:.4f} deg")
    else:
        avg_t_error = 0.0
        avg_r_error = 0.0
        print(f"Val Metrics | Loss: {avg_loss:.4f} (Metrics skipped)")
        
    if name=="val":   
        wandb.log({
            "val/total_loss": avg_loss,
            "val/t_error_m": avg_t_error,
            "val/r_error_deg": avg_r_error,
            "epoch": epoch
        })
    else:
        wandb.log({
            f"{name}/total_loss": avg_loss,
            f"{name}/t_error_m": avg_t_error,
            f"{name}/r_error_deg": avg_r_error,
            "epoch": epoch
        })        
    return avg_loss

if __name__ == "__main__":
    config = {
        "train_csv":  "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/synthetic/train_synthetic.csv",
        "sunlamp_csv" :"/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/sunlamp/sunlamp.csv",
        "lightbox_csv" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/lightbox/lightbox.csv",
        "val_csv" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/synthetic/validation.csv",
        "img_folder" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/",
        "intrinsic_file" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/camera.json",
        "keypoints_file": "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/tangoPoints.mat",
        "save_dir" : "./checkpoints_SEPIA/",
        "batch_size" : 8,  
        "epochs" : 300,
        "lr" : 1e-4,
        "lr_min" : 1e-6,
        "project-name" : "SEPIA_Pose_Estimation",
        "resume_checkpoint": "/home/aac/shared/teams/dtu/mlr-lab/SEPIA/checkpoints_SEPIA/checkpoint_epoch_150.pth",
        "jepa_weights" : "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/logs_cls/jepa_joints_speedplus-latest.pth.tar"
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    wandb_id = None
    if config["resume_checkpoint"]:
        wandb_id = "o43mhue2=="    
    wandb.init(project=config["project-name"], config=config, id=wandb_id, resume="allow")
    train_dataset = SpeedDataset(
        csv_file=config["train_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = SpeedDataset(
        csv_file=config["val_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    sunlamp_dataset = SpeedDataset(
        csv_file=config["sunlamp_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    sunlamp_loader = DataLoader(sunlamp_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    lightbox_dataset = SpeedDataset(
        csv_file=config["lightbox_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    lightbox_loader = DataLoader(lightbox_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    if hasattr(val_dataset, 'intrinsics_mat'):
        K = val_dataset.intrinsics_mat
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    else:
        K = val_dataset.intrinsics
        if torch.is_tensor(K): K = K.numpy()
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])

    
    model = FastPoseViT(
        jepa_path=config["jepa_weights"],
        img_size=224,  # Match the input_size from your dataloader
        patch_size=16
    )
    print("Model Layers")
    print([n for n, p in model.named_parameters()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer_adamw, optimizer_muon = get_optimizer(model, config)
    scheduler_adam = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_adamw, T_0=10, T_mult=2, eta_min=config['lr_min']
    )
    scheduler_muon = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_muon, T_0=10, T_mult=2, eta_min=config['lr_min']
    )
    start_epoch = 0
    best_val_loss = float('inf')

    if config["resume_checkpoint"] and os.path.isfile(config["resume_checkpoint"]):
        print(f"Loading checkpoint: {config['resume_checkpoint']}")
        checkpoint = torch.load(config["resume_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_adamw_state_dict' in checkpoint:
            optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw_state_dict'])
        if 'optimizer_muon_state_dict' in checkpoint:
            optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
        if 'scheduler_adam_state_dict' in checkpoint:
            scheduler_adam.load_state_dict(checkpoint['scheduler_adam_state_dict'])
        if 'scheduler_muon_state_dict' in checkpoint:
            scheduler_muon.load_state_dict(checkpoint['scheduler_muon_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1 
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from Epoch {start_epoch}. Previous Best Val Loss: {best_val_loss:.4f}")
        
    else:
        if config["resume_checkpoint"]:
            print(f"{config['resume_checkpoint']} not found. Starting from scratch.")

    print(f"Starting Training on {device}...")
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_rot_loss = 0.0
        epoch_trans_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(loop):
            images = batch["pixel_values"].to(device)
            R_apparent_gt = batch["R_apparent_gt"].to(device)
            U_gt = batch["U_gt"].to(device)

            optimizer_adamw.zero_grad()
            optimizer_muon.zero_grad()
            
            outputs = model(images)
            U_pred = outputs["U"]
            logits = outputs["logits"]
            pred_deltas = outputs["deltas"]
            anchors = outputs["anchors"]
            
            loss_trans = translation_loss(U_pred, U_gt)
            loss_coarse, loss_fine, coarse_acc, fine_mag = compute_hopf_losses(logits, pred_deltas, anchors, R_apparent_gt)
            
            loss_rot = loss_coarse + (10.0 * loss_fine) # Weight the fine loss
            loss = loss_rot + loss_trans
            
            loss.backward()
            optimizer_adamw.step()
            optimizer_muon.step()
            
            epoch_loss += loss.item()
            epoch_rot_loss += loss_rot.item()
            epoch_trans_loss += loss_trans.item()
            
            loop.set_postfix(loss=loss.item())
            
            wandb.log({
                "train/loss_total": loss.item(),
                "train/loss_trans": loss_trans.item(),
                "train/loss_coarse_rot": loss_coarse.item(),
                "train/loss_fine_rot": loss_fine.item(),
                "diagnostics/coarse_bin_accuracy": coarse_acc.item(),
                "diagnostics/fine_twist_magnitude": fine_mag.item(),
                "lr/adamw": optimizer_adamw.param_groups[0]['lr'],
                "lr/muon": optimizer_muon.param_groups[0]['lr']
            })
        val_loss = validate(model, val_loader, device, val_intrinsics, epoch+1)
        scheduler_adam.step()
        scheduler_muon.step()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_rot_loss = epoch_rot_loss / len(train_loader)
        avg_trans_loss = epoch_trans_loss / len(train_loader)      
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Rot: {avg_rot_loss:.4f}, Trans: {avg_trans_loss:.4f})")
        wandb.log({
            "train/epoch_total_loss": avg_train_loss,
            "train/epoch_rot_loss": avg_rot_loss,
            "train/epoch_trans_loss": avg_trans_loss,
            "epoch": epoch+1,
            "lr_adamw": optimizer_adamw.param_groups[0]['lr'],
            "lr_muon": optimizer_muon.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_adamw_state_dict': optimizer_adamw.state_dict(),
                'scheduler_adam_state_dict': scheduler_adam.state_dict(),
                'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                'scheduler_muon_state_dict': scheduler_muon.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
            print("Evaluating on Sunlamp and Lightbox datasets...")
            print("---------------------------------- Sunlamp Validation --------------------------------")
            _ = validate(model, sunlamp_loader, device, val_intrinsics, epoch+1, name="sunlamp")
            print("---------------------------------- Lightbox Validation--------------------------------#")
            _ = validate(model, lightbox_loader, device, val_intrinsics, epoch+1, name="lightbox")
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_adamw_state_dict': optimizer_adamw.state_dict(),
                'scheduler_adam_state_dict': scheduler_adam.state_dict(), # Added scheduler
                'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                'scheduler_muon_state_dict': scheduler_muon.state_dict(),
                'best_val_loss': best_val_loss,                 # Added best_val_loss
                'loss': avg_train_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Existing local imports from your environment
from dataloader import SpeedDataset
from model import FastPoseViT
from geometry_utils import axis_angle_to_matrix

@torch.no_grad()
def run_jensen_audit_full(config, device, K_samples=16):
    # 1. Setup Dataloader (Exactly as in your train script)
    test_dataset = SpeedDataset(
        csv_file=config["test_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val' # Use 'val' mode to avoid training-specific augmentations
    )
    # Batch size 1 is preferred for per-image audit precision
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. Load Model & Checkpoint
    model = FastPoseViT(img_size=224, patch_size=16).to(device)
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = []

    for batch in tqdm(loader, desc="Running Symmetry Audit"):
        images = batch['pixel_values'].to(device) # Shape: (1, 3, 224, 224)
        
        # 3. Generate K Nuisance Samples (In-plane rotations)
        # This creates the "Orbit" for the audit [cite: 1234]
        angles = torch.linspace(0, 360, K_samples + 1, device=device)[:-1]
        rotated_batch = torch.cat([TF.rotate(images, a.item()) for a in angles])
        
        # 4. Forward Pass through whole pipeline
        out = model(rotated_batch)
        logits = out['logits']
        pred_deltas = out['deltas']
        anchors = out['anchors']
        
        # 5. Recover Apparent Rotations (K, 3, 3)
        winning_bins = torch.argmax(logits, dim=-1)
        R_base = anchors[winning_bins]
        batch_indices = torch.arange(K_samples, device=device)
        delta_win = pred_deltas[batch_indices, winning_bins, :]
        R_delta = axis_angle_to_matrix(delta_win)
        R_preds = torch.bmm(R_base, R_delta) 

        # 6. Coordinate Alignment (Un-rotation)
        # We invert the image-space rotation to test physical constancy [cite: 1104, 1105]
        unrotated_frames = []
        for i, a in enumerate(angles):
            theta = torch.deg2rad(a)
            Rz_inv = torch.eye(3, device=device)
            Rz_inv[0,0], Rz_inv[0,1] = torch.cos(theta), torch.sin(theta)
            Rz_inv[1,0], Rz_inv[1,1] = -torch.sin(theta), torch.cos(theta)
            
            # If the model is invariant, Rz_inv @ R_pred should be constant [cite: 1266]
            unrotated_frames.append(torch.mm(Rz_inv, R_preds[i]))
        
        unrotated_frames = torch.stack(unrotated_frames)

        # 7. Calculate Dispersion (Jensen Gain) [cite: 1236]
        # Measures the variance in the predicted pose across the rotation orbit
        sample_variance = torch.var(unrotated_frames, dim=0).sum().item()
        results.append(sample_variance)

    # Final Statistics
    avg_jensen_gain = np.mean(results)
    print(f"\n[!] Audit Results for {config['test_csv']}")
    print(f"[*] Average Jensen Gain (Dispersion): {avg_jensen_gain:.8f}")
    
    return avg_jensen_gain

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example Config (Point to your test set and best checkpoint)
    audit_config = {
        "test_csv": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/synthetic/validation.csv",
        "img_folder": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/",
        "intrinsic_file": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/camera.json",
        "keypoints_file": "/mnt/external_ssd/ARSH_ARNABI/speedplusv2/tangoPoints.mat",
        "checkpoint_path": "/mnt/external_ssd/ARSH_ARNABI/checkpoint_epoch_220.pth"
    }
    
    run_jensen_audit_full(audit_config, device)
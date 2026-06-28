import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import json
import numpy as np
import os
import scipy.io 
import copy
from PIL import Image
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R_scipy

from ..utils.image_proc import create_belief_map # Assuming this helper exists locally

class SpeedDataset(Dataset):
    def __init__(self, csv_file, image_root, intrinsics_file=None, 
                 keypoints_file='/mnt/external_ssd/ARSH_ARNABI/speedplusv2/tangoPoints.mat', mode='train', crop_size=224):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.mode = mode
        self.train = (mode == 'train')
        self.crop_size = crop_size
        self.epoch = 0
        
        # Image Processing
        self.W_full = 1920
        self.H_full = 1200
        
        # Load Intrinsics (K)
        if intrinsics_file and os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                intrinsics_data = json.load(f)
                self.intrinsics_mat = np.array(intrinsics_data.get('cameraMatrix', 
                    [[2988.57, 0, 960], [0, 2988.34, 600], [0, 0, 1]]), dtype=np.float32)
        else:
            self.intrinsics_mat = np.array([
                [2988.5795, 0, 960],
                [0, 2988.3401, 600],
                [0, 0, 1]
            ], dtype=np.float32)
            
        self.points_3d = self._load_keypoints(keypoints_file)
        
        # Augmentation Settings (Aligning with DREAM)
        self.color_jitter = True
        self.occlusion_augmentation = True
        self.occlu_p = 0.5

    def _load_keypoints(self, filename):
        # ... (Your existing loading logic remains same) ...
        path = filename if os.path.exists(filename) else os.path.join(self.image_root, filename)
        mat = scipy.io.loadmat(path)
        print(f"Keys found in {filename}: {mat.keys()}")
        pts = np.array(mat['tango3Dpoints'], dtype=np.float32) # Standardize to your mat key
        return pts if pts.shape[1] == 3 else pts.T

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.data)

    def _project_to_2d(self, R, T):
        P_cam = np.dot(self.points_3d, R.T) + T
        P_img = np.dot(P_cam, self.intrinsics_mat.T)
        return P_img[:, :2] / P_img[:, 2:3]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_root, row['filename'])
        
        # Load Image
        orig_img_pil = Image.open(img_path).convert('RGB')
        
        # Get Pose
        t_vec = np.array([row['t_x'], row['t_y'], row['t_z']], dtype=np.float32)
        q_vec = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
        r_mat = R_scipy.from_quat(q_vec).as_matrix().astype(np.float32)
        
        # Project Keypoints
        keypoints = self._project_to_2d(r_mat, t_vec) # [N, 2]
        
        # Determine Bounding Box from projected keypoints
        bbox_min = np.min(keypoints, axis=0)
        bbox_max = np.max(keypoints, axis=0)
        
        # Progressive Jitter based on Epoch (Logic from DREAM)
        if self.train:
            jitter_map = {30: 0, 50: 30, 70: 50, 90: 80, 110: 100, 150: 120}
            current_jitter = 120
            for e, j in sorted(jitter_map.items()):
                if self.epoch < e:
                    current_jitter = j
                    break
            
            if current_jitter > 0:
                bbox_min -= np.random.rand(2) * current_jitter
                bbox_max += np.random.rand(2) * current_jitter

        # Clip Bounding Box
        bbox_min = np.clip(bbox_min, [0, 0], [self.W_full, self.H_full])
        bbox_max = np.clip(bbox_max, [0, 0], [self.W_full, self.H_full])

        # Metadata initialization
        metadata = {
            'img_path': img_path,
            'orig_img': np.array(orig_img_pil),
            'orig_keypoints': copy.deepcopy(keypoints),
            'K': self.intrinsics_mat,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max
        }

        # Crop and Resize
        img_np = np.array(orig_img_pil)
        img_crop = img_np[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]
        
        # Handle empty crops (out of bounds)
        if img_crop.size == 0:
            img_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        # Coordinate adjustment for keypoints
        kp_cropped = keypoints.copy()
        kp_cropped[:, 0] -= bbox_min[0]
        kp_cropped[:, 1] -= bbox_min[1]

        # Resizing (Using DREAM-style scaling)
        h_c, w_c = img_crop.shape[:2]
        scale = self.crop_size / max(h_c, w_c)
        new_w, new_h = int(w_c * scale), int(h_c * scale)
        img_resized = cv2.resize(img_crop, (new_w, new_h))
        
        kp_resized = kp_cropped * scale
        
        # Padding to Square
        pad_w = (self.crop_size - new_w) // 2
        pad_h = (self.crop_size - new_h) // 2
        img_final = cv2.copyMakeBorder(img_resized, pad_h, self.crop_size - new_h - pad_h, 
                                       pad_w, self.crop_size - new_w - pad_w, 
                                       cv2.BORDER_CONSTANT, value=0)
        
        kp_final = kp_resized + np.array([pad_w, pad_h])

        # Create Belief Map
        belief_maps = create_belief_map(
            image_resolution=(self.crop_size, self.crop_size), 
            pointsBelief=kp_final, 
            sigma=2
        )
        belief_maps_tensor = torch.from_numpy(belief_maps).float()

        # Image Augmentations (Color/Occlusion)
        if self.train:
            # Here you would call apply_color_jitter, etc., as defined in DREAM's augmentations.py
            pass 

        # Final Tensor Conversion
        img_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img_final)

        # Convert pose to Transformation Matrix (wTc/cTr style)
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r_mat
        pose_mat[:3, 3] = t_vec
        metadata['cTr'] = pose_mat

        # Match DREAM's return signature: img, jointpose, belief_maps, metadata
        # jointpose here is replaced by the raw translation vector or flattened pose
        scale_tuple = (scale, scale)
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = r_mat
        pose_mat[:3, 3] = t_vec
        
        metadata = {
            'img_path': img_path,
            'K': self.intrinsics_mat,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'pad': (pad_w, pad_h),
            'scale': (scale, scale),
            'kp_3d_model': torch.from_numpy(self.points_3d).float(),
            'orig_keypoints_3d': self.points_3d.copy(),
            'cTr': torch.from_numpy(pose_mat).float()  # Ensure this is a tensor
        }
        return img_tensor, torch.from_numpy(t_vec), belief_maps_tensor, metadata

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # 1. Initialize Dataset
    # Replace these paths with your actual local paths
    dataset = SpeedDataset(
        csv_file='/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/synthetic/train_synthetic.csv', 
        image_root='/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/', 
        mode='train'
    )
    
    # Simulate a later epoch to test the progressive jittering
    dataset.set_epoch(80) 
    
    output_dir = 'debug_output'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset size: {len(dataset)}")

    # 2. Extract and Visualize a few samples
    for i in range(min(10, len(dataset))):
        img_tensor, t_vec, belief_maps, metadata = dataset[i]

        # --- STEP 1: De-normalize Image ---
        # Convert from Tensor (C, H, W) to Numpy (H, W, C)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img_vis = img_tensor.permute(1, 2, 0).numpy()
        img_vis = (img_vis * std) + mean
        img_vis = np.clip(img_vis, 0, 1)

        # --- STEP 2: Process Belief Maps ---
        # Collapse all keypoint channels into one [H, W]
        combined_belief = torch.max(belief_maps, dim=0)[0].numpy()
        
        # Apply a colormap to the belief map (Magma or Jet work best)
        # We normalize the belief map to [0, 1] for the colormap
        cm = plt.get_cmap('magma')
        heatmap_colored = cm(combined_belief)[:, :, :3] # Remove alpha channel from CM

        # --- STEP 3: Overlay ---
        # Blend: 70% Image + 30% Heatmap (where heatmap is strong)
        # Or use a simple addition:
        alpha = 0.5
        overlay = (1 - alpha) * img_vis + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        # --- STEP 4: Save & Plot ---
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"Sample {i} - Keypoint Overlay")
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f'overlay_{i}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nFinished! Check the '{output_dir}' folder to verify keypoint alignment")
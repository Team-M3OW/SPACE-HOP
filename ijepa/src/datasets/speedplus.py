import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
import scipy.io
from PIL import Image
from logging import getLogger
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from scipy.spatial.transform import Rotation as R_scipy

# Assuming these utilities exist in your environment
from src.augmentations.aug_utils import FastPoseAugmentations
from src.augmentations.geometry_utils import (
    compute_apparent_rotation, 
    matrix_to_ortho6d, 
    compute_normalized_coords_translation
)

logger = getLogger()

# --- Utilities from DREAM template ---

class ResizeLongerSide:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        h, w = img.shape[:2]
        if w > h:
            new_w = self.crop_size
            new_h = int(self.crop_size * h / w)
        else:
            new_h = self.crop_size
            new_w = int(self.crop_size * w / h)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_img, (new_w, new_h)

class PadToSquare:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        h, w = img.shape[:2]
        pad_h = (self.crop_size - h) // 2
        pad_w = (self.crop_size - w) // 2
        padding = ((pad_h, self.crop_size - h - pad_h), (pad_w, self.crop_size - w - pad_w), (0, 0))
        padded_img = np.pad(img, padding, mode='edge')
        return padded_img, (pad_w, pad_h)

# --- Factory Function ---

def make_speedplus(
    csv_file,
    image_root,
    batch_size,
    transform=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    crop_size=224,
    **kwargs
):
    # Remove drop_last from kwargs if it exists so it doesn't go to SpeedPlus
    kwargs.pop('drop_last', None) 

    dataset = SpeedPlus(
        csv_file=csv_file,
        image_root=image_root,
        transform=transform,
        train=training,
        crop_size=crop_size,
        **kwargs
    )
    
    logger.info(f'SpeedPlus dataset created: {len(dataset)} samples')
    
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training
    )
    
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=training, # Keep drop_last here where it belongs
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    
    return dataset, data_loader, sampler
# --- Main Dataset Class ---

class SpeedPlus(Dataset):
    def __init__(
        self,
        csv_file,
        image_root,
        intrinsics_file=None,
        keypoints_file='/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/tangoPoints.mat',
        transform=None,
        train=True,
        crop_size=224
    ):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.train = train
        self.crop_size = crop_size
        
        # Internal processing components
        self.augmentor = FastPoseAugmentations(train=train)
        self.resize_transform = ResizeLongerSide(crop_size)
        self.pad_transform = PadToSquare(crop_size)
        
        self.W_full = 1920
        self.H_full = 1200
        
        # Load Camera and Keypoint info
        self._setup_camera(intrinsics_file)
        self.points_3d = self._load_keypoints(keypoints_file)

    def _setup_camera(self, intrinsics_file):
        # Default intrinsics
        K = np.array([
            [2988.5795163815555, 0, 960],
            [0, 2988.3401159176124, 600],
            [0, 0, 1]
        ], dtype=np.float32)

        if intrinsics_file and os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                data = json.load(f)
                if 'cameraMatrix' in data:
                    K = np.array(data['cameraMatrix'], dtype=np.float32)
        
        self.intrinsics_mat = K
        self.intrinsics = torch.from_numpy(K)

    def _load_keypoints(self, filename):
        path = filename if os.path.exists(filename) else os.path.join(self.image_root, filename)
        try:
            mat = scipy.io.loadmat(path)
            for k in ['keypoints', 'p_3D', 'tangoPoints', 'points']:
                if k in mat:
                    pts = np.array(mat[k], dtype=np.float32)
                    return pts.T if (pts.shape[0] == 3 and pts.shape[1] > 3) else pts
            return np.array(mat[[k for k in mat.keys() if not k.startswith('__')][0]], dtype=np.float32).T
        except Exception:
            return np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                            [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=np.float32)

    def _project_to_2d(self, R, T):
        P_cam = np.dot(self.points_3d, R.T) + T
        P_img = np.dot(P_cam, self.intrinsics_mat.T)
        z = np.maximum(P_img[:, 2:3], 1e-5)
        return P_img[:, :2] / z

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Image Loading
        filename = row['filename']
        img_path = os.path.join(self.image_root, filename)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.H_full, self.W_full, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Geometry Setup
        t_vec = np.array([row['t_x'], row['t_y'], row['t_z']], dtype=np.float32)
        q_vec = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
        R_cam = R_scipy.from_quat(q_vec).as_matrix().astype(np.float32)

        # 3. Spatial Augmentation (Matches original SpeedPlus flow)
        K = self.intrinsics_mat
        aug_res = self.augmentor.spatial_augmentations(
            img, R_cam, t_vec, intrinsics=(K[0,0], K[1,1], K[0,2], K[1,2])
        )
        img_aug, R_aug, T_aug = aug_res['img_rot'], aug_res['R_new'], aug_res['T_new']

        # 4. Bounding Box and Cropping Logic
        kpts_2d = self._project_to_2d(R_aug, T_aug)
        x_min, y_min = np.min(kpts_2d, axis=0)
        x_max, y_max = np.max(kpts_2d, axis=0)
        
        # Extract tight crop (DREAM style, but with your logic)
        img_crop_raw = img_aug[int(max(0, y_min)):int(min(img_aug.shape[0], y_max)), 
                               int(max(0, x_min)):int(min(img_aug.shape[1], x_max))]
        
        # Handle empty crops
        if img_crop_raw.size == 0:
            img_crop_raw = img_aug

        # 5. Transform Pipeline (DREAM style)
        img_resized, (new_w, new_h) = self.resize_transform(img_crop_raw)
        img_padded, (pad_w, pad_h) = self.pad_transform(img_resized)
        
        # 6. Keypoint Coordinate Transformation
        kpts_tx = kpts_2d.copy()
        kpts_tx[:, 0] -= x_min
        kpts_tx[:, 1] -= y_min
        
        scale_x = new_w / (x_max - x_min) if (x_max - x_min) > 0 else 1.0
        scale_y = new_h / (y_max - y_min) if (y_max - y_min) > 0 else 1.0
        
        kpts_tx[:, 0] = (kpts_tx[:, 0] * scale_x) + pad_w
        kpts_tx[:, 1] = (kpts_tx[:, 1] * scale_y) + pad_h
        # src/datasets/speedplus.py
        # Ensure keypoints stay within the 224x224 boundary
        kpts_tx[:, 0] = np.clip(kpts_tx[:, 0], 0, self.crop_size - 1)
        kpts_tx[:, 1] = np.clip(kpts_tx[:, 1], 0, self.crop_size - 1)
        
        # 7. Final Targets (Pose regression targets)
        R_aug_t = torch.from_numpy(R_aug).T
        T_aug_t = torch.from_numpy(T_aug)
        
        R_apparent = compute_apparent_rotation(R_aug_t, T_aug_t)
        R6D_gt = matrix_to_ortho6d(R_apparent)
        
        # bbox params for U_gt calculation
        crop_size_val = float(max(x_max - x_min, y_max - y_min))
        bbox_params = torch.tensor([(x_min + x_max)/2, (y_min + y_max)/2, crop_size_val, crop_size_val])
        
        U_gt = compute_normalized_coords_translation(
            T_aug_t.unsqueeze(0), bbox_params.unsqueeze(0), 
            (self.W_full, self.H_full), (self.crop_size, self.crop_size), 
            self.intrinsics.unsqueeze(0)
        ).squeeze(0)

        # 8. Pixel Augmentations and Return
        # ... (previous processing logic remains the same) ...

        img_final = Image.fromarray(img_padded)
        if self.transform is not None:
            img_final = self.transform(img_final)
        else:
            img_final = self.augmentor.pixel_augmentations(img_padded)

        # RETURN AS A TUPLE:
        # Index 0: Image (required by collator)
        # Index 1: Keypoints (required by JointMaskCollator for np.random.choice)
        # Index 2: Dictionary of all other metadata/ground truth
        return (
            img_final, 
            torch.from_numpy(kpts_tx).float(),
            {
                'R6D_gt': R6D_gt,
                'U_gt': U_gt,
                'T_gt': T_aug_t,
                'R_gt_true': R_aug_t
            }
        )
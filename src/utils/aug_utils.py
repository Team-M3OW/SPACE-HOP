import albumentations as A
import cv2
import numpy as np
import torch
import random

class FastPoseAugmentations:
    
    def __init__(self, train=True):
        self.train = train
        self.pixel_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
            A.ImageCompression(quality_range=(75, 100), p=0.5),
            A.OpticalDistortion(distort_limit=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def pixel_augmentations(self, img):
        if self.train:
            augmented = self.pixel_transforms(image=img)
            img = augmented['image']
        else:
            normalizer = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            img = normalizer(image=img)['image']
            
        img_tensor = torch.tensor(img).permute(2, 0, 1).float()
        return img_tensor
    
    def bbox_augmentations(self, bbox, img_shape):
        if not self.train:
            return bbox
            
        cx, cy, w, h = bbox
        img_h, img_w = img_shape
        
        d_l, d_r = [random.uniform(-0.1 * w, 0.1 * w) for _ in range(2)]
        d_t, d_b = [random.uniform(-0.1 * h, 0.1 * h) for _ in range(2)]
        
        x1 = max(0, (cx - w / 2) + d_l)
        y1 = max(0, (cy - h / 2) + d_t)
        x2 = min(img_w, (cx + w / 2) + d_r)
        y2 = min(img_h, (cy + h / 2) + d_b)

        new_w, new_h = x2 - x1, y2 - y1
        if new_w <= 1 or new_h <= 1:
            return bbox
        return (x1 + new_w / 2, y1 + new_h / 2, new_w, new_h)
    
    def spatial_augmentations(self, img, R, T, rot_prob=0.5, intrinsics=None):
        if not self.train:
             return {'img_rot': img, 'R_new': R, 'T_new': T}
        angle_deg = 0
        if random.random() < rot_prob:
            if random.random() < 0.5:
                angle_deg = random.uniform(-30, 30)
            else:
                angle_deg = random.uniform(150, 210)
        if angle_deg == 0:
             return {'img_rot': img, 'R_new': R, 'T_new': T}
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R_z = np.array([[c, -s, 0], 
                        [s,  c, 0], 
                        [0,  0, 1]], dtype=np.float32)
        
        R_new = R_z @ R
        T_new = R_z @ T
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        img_rotated = cv2.warpAffine(
            img, M, (w, h), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )

        return {'img_rot': img_rotated, 'R_new': R_new, 'T_new': T_new}
import cv2
import numpy as np
import torch
import cv2
import numpy as np
import torch

def draw_satellite_wireframe(img_tensor, R_gt, T_gt, R_pred, T_pred, K, bbox):
    """
    Draws a 3D wireframe box for Ground Truth (Green) and Prediction (Red)
    overlapping on the same image crop.
    """
    # 1. Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

    # 2. Define a generic 3D bounding box for the Tango satellite (approx bounds in meters)
    l, w, h = 0.35, 0.35, 0.45 
    corners_3d = np.array([
        [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
        [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h]
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
    ]

    # 3. Projection helper function with Crop Adjustment
    def project_and_align_box(R, T):
        R_np = R.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy().reshape(3, 1)
        K_np = K.detach().cpu().numpy()
        
        # 3D to 2D Projection
        P_cam = np.dot(R_np, corners_3d.T) + T_np
        P_img = np.dot(K_np, P_cam)
        pts_2d = P_img[:2, :] / (P_img[2, :] + 1e-6)
        pts_2d = pts_2d.T
        
        # Adjust for the bounding box crop
        bbox_np = bbox.detach().cpu().numpy()
        x1, y1, x2, y2 = bbox_np
        w_box, h_box = (x2 - x1), (y2 - y1)
        _, H_img, W_img = img_tensor.shape
        
        pts_2d[:, 0] = (pts_2d[:, 0] - x1) * (W_img / w_box)
        pts_2d[:, 1] = (pts_2d[:, 1] - y1) * (H_img / h_box)
        return pts_2d.astype(int)

    # 4. Draw Ground Truth (Lime Green)
    gt_uv_box = project_and_align_box(R_gt, T_gt)
    for edge in edges:
        cv2.line(img, tuple(gt_uv_box[edge[0]]), tuple(gt_uv_box[edge[1]]), (0, 255, 0), 2)

    # 5. Draw Prediction (Red)
    pred_uv_box = project_and_align_box(R_pred, T_pred)
    for edge in edges:
        cv2.line(img, tuple(pred_uv_box[edge[0]]), tuple(pred_uv_box[edge[1]]), (0, 0, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
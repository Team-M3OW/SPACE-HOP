import cv2
import numpy as np
import torch

def draw_satellite_wireframe(img_tensor, R_gt, T_gt, R_pred, T_pred, K, bbox):
    # 1. Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()

    # 2. Tango Satellite 3D Bounding Box
    l, w, h = 0.35, 0.35, 0.45 
    corners_3d = np.array([
        [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
        [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h]
    ], dtype=np.float32)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    # 3. Decode [cx, cy, w, h] from your dataloader
    cx, cy, box_w, box_h = bbox.detach().cpu().numpy()
    x1 = cx - (box_w / 2.0)
    y1 = cy - (box_h / 2.0)
    
    _, H_img, W_img = img_tensor.shape # 224x224
    scale_x = W_img / box_w
    scale_y = H_img / box_h

    # 4. Exact Projection math
    def project_and_align_box(R, T):
        R_np = R.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy().reshape(3, 1)
        K_np = K.detach().cpu().numpy()
        
        P_cam = np.dot(R_np, corners_3d.T) + T_np
        P_cam[2, :] = np.clip(P_cam[2, :], a_min=1e-6, a_max=None)
        
        P_img = np.dot(K_np, P_cam)
        uv_raw = (P_img[:2, :] / P_img[2, :]).T
        
        uv_aligned = uv_raw.copy()
        uv_aligned[:, 0] = (uv_aligned[:, 0] - x1) * scale_x
        uv_aligned[:, 1] = (uv_aligned[:, 1] - y1) * scale_y
        
        return uv_aligned.astype(int)

    # 5. Draw
    gt_uv_box = project_and_align_box(R_gt, T_gt)
    for edge in edges: cv2.line(img, tuple(gt_uv_box[edge[0]]), tuple(gt_uv_box[edge[1]]), (0, 255, 0), 2)

    pred_uv_box = project_and_align_box(R_pred, T_pred)
    for edge in edges: cv2.line(img, tuple(pred_uv_box[edge[0]]), tuple(pred_uv_box[edge[1]]), (0, 0, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import torch
import torch.nn.functional as F

"""
geometry_utils.py
Consistent Implementation of FastPose-ViT Geometry
"""

def matrix_to_ortho6d(R):
    # Args: R: (B, 3, 3) or (3, 3)
    # Returns: (B, 6) or (6,)
    if R.dim() == 3:
        r1 = R[:, :, 0] 
        r2 = R[:, :, 1]
        return torch.cat([r1, r2], dim=1)
    else:
        r1 = R[:, 0]
        r2 = R[:, 1]
        return torch.cat([r1, r2], dim=0)
    
def ortho6d_to_matrix(ortho6d):
    # Args: ortho6d: (B, 6)
    # Returns: R: (B, 3, 3)
    r1 = ortho6d[:, :3]
    r2 = ortho6d[:, 3:]
    
    r1_norm = F.normalize(r1, dim=-1)
    dot = torch.sum(r1_norm * r2, dim=-1, keepdim=True)
    u2 = r2 - dot * r1_norm
    r2_norm = F.normalize(u2, dim=-1)
    r3_norm = torch.cross(r1_norm, r2_norm, dim=-1)
    
    R_matrix = torch.stack([r1_norm, r2_norm, r3_norm], dim=-1)
    return R_matrix

def compute_delta_rotation(T_batch, device):
    """
    Computes the rotation matrix Delta_R that aligns the Z-axis to T.
    """
    B = T_batch.shape[0]
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).view(1, 3).expand(B, 3)
    
    T_norm = F.normalize(T_batch, dim=-1)
    
    cross_prod = torch.cross(T_norm, ez, dim=-1) # Note: order matters for sign
    cross_norm = torch.norm(cross_prod, dim=-1, keepdim=True)
    mask_singular = cross_norm.squeeze() < 1e-6
    
    u = torch.zeros_like(T_norm)
    if not mask_singular.all():
         u[~mask_singular] = cross_prod[~mask_singular] / cross_norm[~mask_singular]
    
    dot_prod = torch.sum(T_norm * ez, dim=-1, keepdim=True)
    theta = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    
    zero = torch.zeros(B, device=device)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    u_skew = torch.stack([zero, -uz, uy, uz, zero, -ux, -uy, ux, zero], dim=1).reshape(B, 3, 3)
    
    I = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    
    Delta_R = I + sin_theta * u_skew + (1.0 - cos_theta) * torch.bmm(u_skew, u_skew)
    
    if mask_singular.any():
        Delta_R[mask_singular] = torch.eye(3, device=device)
        
    return Delta_R

def compute_apparent_rotation(R_true, T_true):
    """
    Target Generation: R' = Delta_R @ R_true
    """
    if R_true.dim() == 2: R_true = R_true.unsqueeze(0)
    if T_true.dim() == 1: T_true = T_true.unsqueeze(0)
    
    Delta_R = compute_delta_rotation(T_true, R_true.device)
    
    # Paper Equation 8
    R_prime = torch.bmm(Delta_R, R_true)
    
    return R_prime.squeeze(0) if R_true.shape[0] == 1 else R_prime

def recover_true_rotation(R_prime_pred, T_pred):
    """
    Inference Recovery: R = Delta_R.T @ R'
    """
    Delta_R = compute_delta_rotation(T_pred, T_pred.device)  
    
    # Paper Equation 10 (Transpose is Inverse for Rotation)
    Delta_R_inv = Delta_R.transpose(1, 2)
    
    R_pred = torch.bmm(Delta_R_inv, R_prime_pred)
    
    return R_pred

def compute_normalized_coords_translation(T_true, bbox, full_img_size, crop_size, intrinsics, alpha=1.6):
    B = T_true.shape[0]
    W_full, H_full = full_img_size
    
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx_cam = intrinsics[:, 0, 2]
    cy_cam = intrinsics[:, 1, 2]

    X, Y, Z = T_true[:, 0], T_true[:, 1], T_true[:, 2]
    bx, by, bw, bh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    sx = W_full / (alpha * bw)
    sy = H_full / bh

    Uz = 0.5 * ((1.0 / sx) + (1.0 / sy)) * Z
    Ux = ((fx / Z) * X + cx_cam - bx) / bw
    Uy = ((fy / Z) * Y + cy_cam - by) / bh

    U = torch.stack([Ux, Uy, Uz], dim=1)
    return U

def recover_translation_components(U_pred, bbox, full_img_size, crop_size, intrinsics, alpha=1.6):
    W_full, H_full = full_img_size
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx_cam = intrinsics[:, 0, 2]
    cy_cam = intrinsics[:, 1, 2]

    bx, by, bw, bh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    sx = W_full / (alpha * bw)
    sy = H_full / bh
    
    Ux_pred, Uy_pred, Uz_pred = U_pred[:, 0], U_pred[:, 1], U_pred[:, 2]

    Z = 2 * Uz_pred / ((1.0 / sx) + (1.0 / sy))
    X = (bx + Ux_pred * bw - cx_cam) * (Z / fx)
    Y = (by + Uy_pred * bh - cy_cam) * (Z / fy)

    T = torch.stack([X, Y, Z], dim=1)
    return T
import torch
import numpy as np

def generate_fibonacci_sphere(num_points, device='cpu'):
    """
    Generates approximately uniform points on a 2D sphere (S^2) using a Fibonacci lattice.
    """
    indices = torch.arange(0, num_points, dtype=torch.float32, device=device)
    phi = torch.arccos(1 - 2 * (indices + 0.5) / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)  # Shape: [N, 3]

def _create_frame_from_z_vectorized(z_vecs):
    """
    Constructs valid 3x3 rotation matrices where the Z-axis aligns with z_vecs.
    Fully vectorized to avoid Python loops.
    """
    # Normalize Z vectors
    z_vecs = torch.nn.functional.normalize(z_vecs, dim=-1)
    
    # Base UP vector broadcasted to match z_vecs shape
    up = torch.tensor([0.0, 0.0, 1.0], device=z_vecs.device, dtype=torch.float32).expand_as(z_vecs).clone()
    
    # Hairy Ball Theorem discontinuity handling (Poles)
    pole_mask = torch.abs(z_vecs[:, 2]) > 0.999
    up[pole_mask] = torch.tensor([1.0, 0.0, 0.0], device=z_vecs.device, dtype=torch.float32)
        
    x_vecs = torch.linalg.cross(up, z_vecs)
    x_vecs = torch.nn.functional.normalize(x_vecs, dim=-1)
    
    y_vecs = torch.linalg.cross(z_vecs, x_vecs)
    y_vecs = torch.nn.functional.normalize(y_vecs, dim=-1)
    
    # Stack along the last dimension to form columns [X, Y, Z]
    return torch.stack([x_vecs, y_vecs, z_vecs], dim=-1) # Shape: [N, 3, 3]

def generate_hopf_so3_grid(num_points=256, num_rolls=12, device='cpu'):
    """
    Generates a uniform deterministic grid on SO(3) using the Hopf Fibration principle.
    Fully vectorized.
    Returns:
        anchors: Tensor of shape [num_points * num_rolls, 3, 3]
    """
    # 1. Generate uniform S^2 base pointing directions
    z_dirs = generate_fibonacci_sphere(num_points, device=device)
    R_base = _create_frame_from_z_vectorized(z_dirs) # [N, 3, 3]
    
    # 2. Generate uniform S^1 roll angles (exclude 2pi)
    roll_angles = torch.linspace(0, 2 * np.pi, num_rolls + 1, device=device)[:-1]
    
    c = torch.cos(roll_angles)
    s = torch.sin(roll_angles)
    
    # 3. Build Roll Matrices
    R_roll = torch.zeros((num_rolls, 3, 3), device=device, dtype=torch.float32)
    R_roll[:, 0, 0] = c
    R_roll[:, 0, 1] = -s
    R_roll[:, 1, 0] = s
    R_roll[:, 1, 1] = c
    R_roll[:, 2, 2] = 1.0
    
    # 4. Compose rotations via broadcasting matrix multiplication
    # [N, 1, 3, 3] @ [1, R, 3, 3] -> [N, R, 3, 3]
    R_final = torch.matmul(R_base.unsqueeze(1), R_roll.unsqueeze(0))
            
    return R_final.view(-1, 3, 3) # Flatten to [N * R, 3, 3]

def get_closest_anchor(gt_pose, anchors):
    """
    Finds the index of the closest SO(3) anchor for a batch of ground truth rotations.
    Utilizes parallelized Einstein summation for trace maximization.
    """
    # trace(R_gt * R_anchor^T) -> b: batch, i/j: 3x3 matrix dims, k: anchor grid
    traces = torch.einsum('bij,kij->bk', gt_pose, anchors)
    return torch.argmax(traces, dim=1)

if __name__ == "__main__":
    print("Testing Vectorized Hopf Fibration SO(3) Grid Generator...")
    
    anchors = generate_hopf_so3_grid(num_points=256, num_rolls=12, device='cpu')
    print(f"Generated Grid Shape: {anchors.shape} (Expected: [3072, 3, 3])")
    
    # Orthogonality Check
    I_pred = torch.matmul(anchors[0], anchors[0].transpose(0, 1))
    assert torch.allclose(I_pred, torch.eye(3), atol=1e-5), "Matrices are not orthogonal!"
    print("Orthogonality check passed.")
    
    # Timing / Einsum test
    import time
    gt_batch = torch.stack([anchors[15], anchors[1200], anchors[3000]])
    t0 = time.time()
    closest_indices = get_closest_anchor(gt_batch, anchors)
    t1 = time.time()
    
    print(f"Closest Indices: {closest_indices.tolist()} (Expected: [15, 1200, 3000])")
    print(f"Einsum Batch Match completed in {(t1-t0)*1000:.2f} ms")
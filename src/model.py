import torch
import torch.nn as nn
import os
import sys

# Inject your custom JEPA path
jepa_root = "/mnt/external_ssd/ARSH_ARNABI/ijepa"
if jepa_root not in sys.path:
    sys.path.append(jepa_root)

from src.models.vision_transformer import vit_base
from hopf_grid import generate_hopf_so3_grid

class FastPoseViT(nn.Module):
    def __init__(self, jepa_path="/mnt/external_ssd/ARSH_ARNABI/SEPIA/jepa_joints_speedplus-ep300.pth.tar", img_size=224, patch_size=16):
        super(FastPoseViT, self).__init__()
        
        # 1. YOUR JEPA BACKBONE (Preserving your custom pre-training)
        self.vit = vit_base(img_size=[img_size], patch_size=patch_size)
        embed_dim = 768 
        
        # Load custom SPEED+ JEPA weights
        if jepa_path and os.path.exists(jepa_path):
            print(f"[*] Loading custom JEPA backbone from: {jepa_path}")
            checkpoint = torch.load(jepa_path, map_location='cpu')
            
            if 'target_encoder' in checkpoint: state_dict = checkpoint['target_encoder']
            elif 'encoder' in checkpoint: state_dict = checkpoint['encoder']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            else: state_dict = checkpoint
                
            clean_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '').replace('encoder.', '').replace('target_encoder.', '')
                clean_dict[new_k] = v
                
            msg = self.vit.load_state_dict(clean_dict, strict=False)
            print(f"[*] JEPA Load Message: {msg}")
        
        # Generate the discrete Hopf anchors
        anchors = generate_hopf_so3_grid(num_points=256, num_rolls=12, device='cpu')
        self.register_buffer('anchors', anchors)
        self.K_bins = anchors.shape[0]

        # -------------------------------------------------------------------
        # 2. THE DITTO FASTPOSE HEAD (Predicts 9D)
        # -------------------------------------------------------------------
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, 9)
        )
        nn.init.xavier_uniform_(self.pose_head[0].weight)
        nn.init.zeros_(self.pose_head[0].bias)

        # -------------------------------------------------------------------
        # 3. COARSE: CROSS-ATTENTION WITH CLS
        # -------------------------------------------------------------------
        # CLS token queries the spatial patches to build a rotation-aware feature
        self.coarse_cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.coarse_classifier = nn.Linear(embed_dim, self.K_bins)
        
        # -------------------------------------------------------------------
        # 4. FINE: BIN-CONSTRAINED TWIST
        # -------------------------------------------------------------------
        self.fine_head = nn.Linear(embed_dim, self.K_bins * 3)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

    def forward(self, x):
        # Raw JEPA outputs: (B, 197, D)
        outputs = self.vit(x)
        
        # Split CLS from patches for Cross-Attention
        cls_token = outputs[:, 0:1, :]   # Shape: (B, 1, D)
        patch_tokens = outputs[:, 1:, :] # Shape: (B, 196, D)
        
        # --- TRANSLATION (FastPose Ditto) ---
        # Predict all 9, slice out the first 3 for Translation $U$
        pose_9d = self.pose_head(cls_token.squeeze(1))
        U_pred = pose_9d[:, :3] 
        
        # --- COARSE (Cross-Attention) ---
        # CLS token acts as Query; Patches act as Key/Value
        attn_out, _ = self.coarse_cross_attn(query=cls_token, key=patch_tokens, value=patch_tokens)
        attn_out = attn_out.squeeze(1) # Shape back to (B, D)
        logits = self.coarse_classifier(attn_out)
        
        # --- FINE (Bin Constrained via Skip Connection) ---
        deltas = self.fine_head(cls_token.squeeze(1)).view(-1, self.K_bins, 3)
        
        return {
            'U': U_pred,
            'logits': logits,
            'deltas': deltas,
            'anchors': self.anchors
        }
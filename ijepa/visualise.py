import os
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.decomposition import PCA
from torchvision import transforms
from src.datasets.speedplus import SpeedPlus
from src.models.vision_transformer import vit_base


# ----------------------------
# Robust Model Loading
# ----------------------------

def load_ijepa_encoder(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['encoder'] if 'encoder' in checkpoint else checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Successfully loaded model from {checkpoint_path}")
    return model


# ----------------------------
# Attention Rollout
# ----------------------------

def compute_rollout(attentions, discard_ratio=0.8):
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            fused = attention.mean(axis=1)[0]

            flat = fused.view(-1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
            fused.view(-1)[indices] = 0

            I = torch.eye(fused.size(-1))
            a = (fused + I) / 2
            a = a / a.sum(dim=-1).unsqueeze(-1)

            result = torch.matmul(a, result)

    return result


# ----------------------------
# Visualizer
# ----------------------------

class IJEPAVisualizer:
    def __init__(self, model, device, patch_size=16):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.attentions = []
        self.features = []
        self._register_hooks()

    def _register_hooks(self):

        def attn_hook(module, input, output):
            self.attentions.append(output[1].detach().cpu())

        def feat_hook(module, input, output):
            self.features.append(output.detach().cpu())

        for block in self.model.blocks:
            block.attn.register_forward_hook(attn_hook)

        self.model.norm.register_forward_hook(feat_hook)

    def unnormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor.cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

    def run_analysis(self, dataset, csv_path, n_samples=10,
                     parent_output_dir=".", subfolder_name="analysis_outputs"):

        output_root = os.path.join(parent_output_dir, subfolder_name)
        os.makedirs(output_root, exist_ok=True)

        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        indices = random.sample(range(len(dataset)), n_samples)

        print(f"Saving outputs to: {os.path.abspath(output_root)}")

        for idx in indices:

            sample_folder = f"{csv_name}_{idx}"
            base_path = os.path.join(output_root, sample_folder)
            os.makedirs(base_path, exist_ok=True)

            img_tensor, kpts, meta = dataset[idx]
            self.attentions = []
            self.features = []

            with torch.no_grad():
                _ = self.model(img_tensor.unsqueeze(0).to(self.device))

            img_show = self.unnormalize(img_tensor)

            # ---------------- ORIGINAL ----------------
            fig1, ax1 = plt.subplots()
            ax1.imshow(img_show)
            ax1.scatter(kpts[:, 0], kpts[:, 1], c='lime', s=15, marker='x')
            ax1.set_title(f"Z: {meta['T_gt'][2]:.2f}m")
            ax1.axis('off')
            fig1.savefig(os.path.join(base_path, "original.png"), dpi=300)
            plt.close(fig1)

            # ---------------- ATTENTION ----------------
            rollout_mat = compute_rollout(self.attentions)
            mask = rollout_mat[0, :].cpu().numpy()

            grid_h = img_tensor.shape[1] // self.patch_size
            grid_w = img_tensor.shape[2] // self.patch_size

            if len(mask) > (grid_h * grid_w):
                mask = mask[1:]

            mask = mask.reshape(grid_h, grid_w)
            mask = cv2.resize(mask / (mask.max() + 1e-8),
                              (img_tensor.shape[2], img_tensor.shape[1]))

            fig2, ax2 = plt.subplots()
            ax2.imshow(img_show)
            ax2.imshow(mask, cmap='jet', alpha=0.5)
            ax2.axis('off')
            fig2.savefig(os.path.join(base_path, "attention.png"), dpi=300)
            plt.close(fig2)

            # ---------------- PCA FEATURE MAP ----------------
            feats = self.features[-1].squeeze().cpu().numpy()

            if feats.shape[0] > (grid_h * grid_w):
                feats = feats[1:]

            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feats)

            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            feat_pca = feat_pca.reshape(grid_h, grid_w, 3)
            feat_pca = cv2.resize(feat_pca,
                                  (img_tensor.shape[2], img_tensor.shape[1]))

            fig3, ax3 = plt.subplots()
            ax3.imshow(feat_pca)
            ax3.axis('off')
            fig3.savefig(os.path.join(base_path, "pca_feature.png"), dpi=300)
            plt.close(fig3)

            # ---------------- GRID ----------------
            fig4, axes = plt.subplots(2, 2, figsize=(8, 8))

            axes[0, 0].imshow(img_show)
            axes[0, 0].scatter(kpts[:, 0], kpts[:, 1], c='lime', s=10)
            axes[0, 0].set_title("Original + Kpts")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(img_show)
            axes[0, 1].imshow(mask, cmap='jet', alpha=0.5)
            axes[0, 1].set_title("Attention")
            axes[0, 1].axis('off')

            axes[1, 0].imshow(feat_pca)
            axes[1, 0].set_title("PCA Feature")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(img_show)
            axes[1, 1].set_title(f"Z = {meta['T_gt'][2]:.2f}m")
            axes[1, 1].axis('off')

            plt.tight_layout()
            fig4.savefig(os.path.join(base_path, "grid.png"), dpi=300)
            plt.close(fig4)

            print(f"Saved sample {idx} → {base_path}")


# ----------------------------
# Execution Block
# ----------------------------

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PARENT_OUTPUT_DIR = "/home/aac/shared/teams/dtu/mlr-lab/lightbox_vis2/"

    CSV_PATH = '/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/lightbox/lightbox.csv'

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SpeedPlus(
        csv_file=CSV_PATH,
        image_root='/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/',
        transform=data_transform,
        train=False,
        crop_size=224
    )

    model = vit_base(img_size=[224], patch_size=16)

    CKPT_PATH = "/home/aac/shared/teams/dtu/mlr-lab/speedplusv2/logs_cls/jepa_joints_speedplus-ep300.pth.tar"
    model = load_ijepa_encoder(model, CKPT_PATH, DEVICE)

    viz = IJEPAVisualizer(model, DEVICE)
    viz.run_analysis(
        dataset,
        csv_path=CSV_PATH,
        n_samples=50,
        parent_output_dir=PARENT_OUTPUT_DIR
    )
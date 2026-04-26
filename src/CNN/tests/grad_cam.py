from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.CNN.models.models import LateFusionWithAttention

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')


class SingleBranchWrapper(torch.nn.Module):
    """
    Wraps one branch of LateFusionWithAttention so GradCAM
    can target it independently.
    Input: (B, 3, H, W) — single subband
    Output: (B, 2) — uses full model head with other branches zeroed
    """
    def __init__(self, model, branch_idx):
        super().__init__()
        self.model      = model
        self.branch_idx = branch_idx

    def forward(self, x):
        # Run only the target branch, zero all others
        feats = []
        for i in range(self.model.n_subbands):
            if i == self.branch_idx:
                feat = self.model.branches[i](x)
            else:
                # Detached zeros — same shape as real features
                with torch.no_grad():
                    feat = torch.zeros(
                        x.shape[0], 64,
                        device=x.device, dtype=x.dtype
                    ).detach()
            feats.append(feat)

        stacked = torch.stack(feats, dim=1)        # (B, n_sub, 64)
        flat    = stacked.reshape(stacked.size(0), -1)
        weights = self.model.attention(flat).unsqueeze(-1)
        fused   = (stacked * weights).sum(dim=1)
        return self.model.head(fused)



def compute_gradcam_per_subband(model, x_multi, target_class=1,
                                 subband_names=None, device='mps'):
    """
    Compute Grad-CAM for each subband branch independently.

    Args:
        model:         trained LateFusionWithAttention
        x_multi:       (1, 21, 256, 256) single sample tensor
        target_class:  1=Pre-AF, 0=NSR
        subband_names: ['D2','D3',...,'D8']

    Returns:
        cams: dict {subband_name: (256, 256) heatmap float32}
    """
    if subband_names is None:
        subband_names = [f'D{i+2}' for i in range(model.n_subbands)]

    model.eval()
    cams = {}

    for i, name in enumerate(subband_names):
        # Extract this subband's 3 channels
        x_sub = x_multi[:, i*3:(i+1)*3, :, :]   # (1, 3, 256, 256)

        # Wrap model to target this branch
        wrapper = SingleBranchWrapper(model, branch_idx=i).to(device)

        # Target the last conv layer of this branch
        # In make_backbone: index 8 is the last Conv2d before GAP
        target_layer = [model.branches[i][8]]

        cam = GradCAM(model=wrapper, target_layers=target_layer)

        targets = [lambda logits: logits[:, target_class]]

        grayscale_cam = cam(
            input_tensor=x_sub.to(device),
            targets=None,   # None = target the predicted class
        )   # (1, H, W)

        cams[name] = grayscale_cam[0]   # (H, W)

    return cams


def plot_gradcam_grid(x_multi, cams, labels_true,
                      subband_names, sample_idx=0,
                      save_path=None):
    """
    Plot original UTRP + Grad-CAM overlay for each subband.
    Rows: NSR sample, Pre-AF sample
    Columns: subbands D2-D8
    """
    n_sub = len(subband_names)
    fig, axes = plt.subplots(2, n_sub, figsize=(n_sub * 2.5, 5))

    for col, name in enumerate(subband_names):
        for row, (x, cam, label) in enumerate(zip(x_multi, cams, labels_true)):
            # Original image (H, W, 3) in [0,1]
            img = x[col*3:(col+1)*3].permute(1, 2, 0).cpu().numpy()

            # Overlay
            overlay = show_cam_on_image(img, cam[name], use_rgb=True)

            axes[row, col].imshow(overlay)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(name, fontsize=9, fontweight='bold')
            if col == 0:
                class_str = 'Pre-AF' if label == 1 else 'NSR'
                colour    = '#d62728' if label == 1 else '#1f77b4'
                axes[row, col].set_ylabel(class_str, fontsize=9,
                                           color=colour, fontweight='bold')

    plt.suptitle('Grad-CAM per Subband — Pre-AF vs NSR', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    else:
        plt.show()
    plt.close()

def load_base_model(ckpt_path: Path, n_subbands: int = 7,
                     dropout: float = 0.5) -> LateFusionWithAttention:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = LateFusionWithAttention(n_subbands=n_subbands,
                                    dropout=dropout).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model



if __name__ == '__main__':
    from src.CNN.dataset.dataset import UTRPDataset

    model_path = Path('/results_new/multi/final_checkpoint.pt')
    data_dir = Path('/data_final/utrp_dataset_db8_no_denoise')
    results_dir = Path('/results_new/grad_cam')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load one NSR and one Pre-AF sample



    test_ds = UTRPDataset(data_dir, split='test', mode='multi', augment=False)
    labels = test_ds.labels

    nsr_idx = np.where(labels == 0)[0][0]
    preaf_idx = np.where(labels == 1)[0][0]

    x_nsr, _ = test_ds[nsr_idx]
    x_preaf, _ = test_ds[preaf_idx]

    # Stack to batch of 2
    x_batch = torch.stack([x_nsr, x_preaf]).to(DEVICE)

    # Compute Grad-CAM for each sample
    model = load_base_model(model_path)

    cams_nsr = compute_gradcam_per_subband(model, x_nsr.unsqueeze(0).to(DEVICE))
    cams_preaf = compute_gradcam_per_subband(model, x_preaf.unsqueeze(0).to(DEVICE))

    # Combine for plotting
    all_cams = [cams_nsr, cams_preaf]

    plot_gradcam_grid(
        x_batch,
        all_cams,
        labels_true=[0, 1],
        subband_names=['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
        save_path=results_dir / 'gradcam_per_subband.png'
    )


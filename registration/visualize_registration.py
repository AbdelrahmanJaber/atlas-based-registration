import os
import torch
import matplotlib.pyplot as plt
from models.voxelmorph.dataset import NiftiPairDataset
from models.voxelmorph.model import VoxelMorphNet
import yaml

# Load config
with open('configs/baseline_voxelmorph.yaml', 'r') as f:
    config = yaml.safe_load(f)

def plot_registration(fixed, moving, warped, out_path, slice_idx=None):
    # fixed, moving, warped: (1, D, H, W) torch tensors
    fixed = fixed.squeeze().cpu().numpy()
    moving = moving.squeeze().cpu().numpy()
    warped = warped.squeeze().cpu().numpy()
    D = fixed.shape[0]
    if slice_idx is None:
        slice_idx = D // 2  # Middle slice
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(fixed[slice_idx], cmap='gray')
    axs[0].set_title('Fixed')
    axs[1].imshow(moving[slice_idx], cmap='gray')
    axs[1].set_title('Moving')
    axs[2].imshow(warped[slice_idx], cmap='gray')
    axs[2].set_title('Warped Moving')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load dataset (use test or val split)
    dataset = NiftiPairDataset(
        data_root=config['data_root'],
        split_file=config['split_file'],
        split='test' if 'test' in config['split_file'] else 'val'
    )
    # Load model
    img_size = dataset[0][0].shape[1:]  # (D, H, W)
    model = VoxelMorphNet(
        in_channels=2,
        unet_features=config['model']['unet_features'],
        img_size=img_size
    ).to(device)
    # Find latest checkpoint
    ckpts = [f for f in os.listdir(config['output_dir']) if f.endswith('.pt')]
    if not ckpts:
        raise FileNotFoundError('No model checkpoint found in output_dir')
    latest_ckpt = sorted(ckpts)[-1]
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], latest_ckpt), map_location=device))
    model.eval()
    # Pick a random pair
    moving, fixed = dataset[0]
    moving = moving.unsqueeze(0).to(device)
    fixed = fixed.unsqueeze(0).to(device)
    with torch.no_grad():
        warped, flow = model(moving, fixed)
    # Visualize
    out_path = os.path.join(config['output_dir'], 'registration_example.png')
    plot_registration(fixed[0], moving[0], warped[0], out_path)
    print(f'Saved visualization to {out_path}')

if __name__ == '__main__':
    main() 
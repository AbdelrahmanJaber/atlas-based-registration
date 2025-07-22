import os
import yaml
import torch
from torch.utils.data import DataLoader
from models.voxelmorph.dataset import NiftiPairDataset
from models.voxelmorph.model import VoxelMorphNet
import torch.nn as nn
import torch.optim as optim

# Load config
with open('configs/baseline_voxelmorph.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_dataloader(split):
    dataset = NiftiPairDataset(
        data_root=config['data_root'],
        split_file=config['split_file'],
        split=split
    )
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 4))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training and validation dataloaders
    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val') if 'val' in config['split_file'] or os.path.exists(config['split_file']) else None

    # Model
    img_size = next(iter(train_loader))[0].shape[2:]  # (D, H, W)
    model = VoxelMorphNet(
        in_channels=2,
        unet_features=config['model']['unet_features'],
        img_size=img_size
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    os.makedirs(config['output_dir'], exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.0
        for i, (moving, fixed) in enumerate(train_loader):
            moving = moving.to(device)
            fixed = fixed.to(device)
            optimizer.zero_grad()
            warped_moving, flow = model(moving, fixed)
            loss = criterion(warped_moving, fixed)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % config.get('log_interval', 10) == 0:
                print(f"Epoch [{epoch}/{config['epochs']}], Step [{i+1}], Loss: {loss.item():.4f}")
        avg_loss = running_loss / (i + 1)
        print(f"Epoch [{epoch}/{config['epochs']}], Average Train Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for j, (moving, fixed) in enumerate(val_loader):
                    moving = moving.to(device)
                    fixed = fixed.to(device)
                    warped_moving, flow = model(moving, fixed)
                    loss = criterion(warped_moving, fixed)
                    val_loss += loss.item()
            avg_val_loss = val_loss / (j + 1)
            print(f"Epoch [{epoch}/{config['epochs']}], Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if epoch % config.get('save_every', 10) == 0 or epoch == config['epochs']:
            checkpoint_path = os.path.join(config['output_dir'], f"voxelmorph_epoch{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == '__main__':
    main()

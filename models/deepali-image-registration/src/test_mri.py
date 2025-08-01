import os

import logging
import datetime
import configparser
import argparse

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision import transforms

from deepali.core.environ import cuda_visible_devices

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

from classes.metrics import ImageMetrics
from classes.model import VoxelMorph
from classes.own_dataset import CustomDataset
from classes.own_dataset_json import CustomDataset as CustomDatasetJson
from classes.losses import VoxelMorphLoss, VoxelMorphDataLoss, VoxelMorphSegLoss
from tqdm import tqdm
import random


log_dir = "deepali_vxl/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # TensorBoard log directory
writer = SummaryWriter(log_dir=log_dir) # Create a SummaryWriter object

def custom_split(indices, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Splittet eine Liste von Indizes in train/val/test.
    Die Werte müssen zusammen 1 ergeben.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    random.seed(seed)
    indices = list(indices)
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]
    test_idx  = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

def test(model, test_loader, loss_func, loss_weight, metrics, device, output_dir, grid, seg=False):
    '''
    Test the model with input shape [200, 640, 640] (gshape),
    process each slice ([1, 640, 640]) separately, concatenate results,
    and save everything as npy.
    '''

    model.eval()
    save_dir = "evaluation_outputs"
    os.makedirs(save_dir, exist_ok=True)

    all_atlas = []
    all_warped = []
    all_moving = []
    all_flows = []
    if seg:
        all_atlas_segs = []
        all_warped_segs = []
        all_moving_segs = []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', unit='batch'):
            print(f"Testing batch {batch_idx + 1}/{len(test_loader)}")
            # -> Erwartet: data[0] shape: [200, 640, 640] (z.B. source)
            # Falls seg: data = (source, target, source_seg, target_seg)
            if seg:
                source, target, source_seg, target_seg = data
            else:
                source, target = data

            source = source.squeeze(0).to(device)
            target = target.squeeze(0).to(device) 
            if seg:
                source_seg = source_seg.squeeze(0).to(device)
                target_seg = target_seg.squeeze(0).to(device)

            num_slices = source.shape[0]  # 200

            batch_atlas = []
            batch_warped = []
            batch_moving = []
            batch_flows = []
            if seg:
                batch_atlas_segs = []
                batch_warped_segs = []
                batch_moving_segs = []

            print(f"Number of slices in batch: {num_slices}")

            for i in range(num_slices):
                s = source[i].unsqueeze(0).unsqueeze(0).to(device)     # [1, 1, 640, 640]
                t = target[i].unsqueeze(0).unsqueeze(0).to(device)     # [1, 1, 640, 640]

                print(f"Shape source: {s.shape}, Shape target: {t.shape}")
                if seg:
                    ss = source_seg[i].unsqueeze(0).unsqueeze(0).to(device)
                    ts = target_seg[i].unsqueeze(0).unsqueeze(0).to(device)
                    # Model expects ((source, source_seg), target)
                    print(f"Shape source: {s.shape}, Shape target: {t.shape}, Shape source_seg: {ss.shape}, Shape target_seg: {ts.shape}")
                    transformed, flow, transformed_seg = model((s, ss), t)
                else:
                    transformed, flow = model(s, t)

                # to numpy
                batch_atlas.append(s.squeeze().cpu().numpy())
                batch_warped.append(transformed.squeeze().cpu().numpy())
                batch_moving.append(t.squeeze().cpu().numpy())
                # flow shape [1, 2, 640, 640] -> [2, 640, 640]
                batch_flows.append(flow.squeeze().cpu().numpy())

                if seg:
                    batch_atlas_segs.append(ss.squeeze().cpu().numpy())
                    batch_warped_segs.append(transformed_seg.squeeze().cpu().numpy())
                    batch_moving_segs.append(ts.squeeze().cpu().numpy())

            # Nach jedem Batch alles in das große Array
            all_atlas.append(np.stack(batch_atlas))        # [200, 640, 640]
            all_warped.append(np.stack(batch_warped))
            all_moving.append(np.stack(batch_moving))
            all_flows.append(np.stack(batch_flows))        # [200, 2, 640, 640]
            if seg:
                all_atlas_segs.append(np.stack(batch_atlas_segs))
                all_warped_segs.append(np.stack(batch_warped_segs))
                all_moving_segs.append(np.stack(batch_moving_segs))

    # Endgültig alles zusammenbauen: [num_batches, 200, ...] -> [-1, ...]
    atlas = np.concatenate(all_atlas, axis=0)           # [N, 640, 640]
    warped = np.concatenate(all_warped, axis=0)
    moving = np.concatenate(all_moving, axis=0)
    flows = np.concatenate(all_flows, axis=0)           # [N, 2, 640, 640]
    if seg:
        atlas_segs = np.concatenate(all_atlas_segs, axis=0)
        warped_segs = np.concatenate(all_warped_segs, axis=0)
        moving_segs = np.concatenate(all_moving_segs, axis=0)

    # Save
    np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_atlas.npy"), atlas)
    np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_warped.npy"), warped)
    np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_moving.npy"), moving)
    np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_flow.npy"), flows)
    if seg:
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_atlas_seg.npy"), atlas_segs)
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_warped_seg.npy"), warped_segs)
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_run_moving_segs.npy"), moving_segs)

def visualize_flow(flow):
    '''
    Visualisiert den Flussbetrag für alle Bilder im Batch oder für Einzelbilder.

    :param flow: Flow tensor oder numpy array
    :return: Flow magnitude tensor
    '''
    # Falls numpy array: in torch tensor umwandeln
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)

    if flow.ndim == 3:
        flow = flow.unsqueeze(0)  # zu (1, 2, H, W)

    print(f"Flow shape: {flow.shape}")

    batch_size = flow.shape[0]
    magnitude_list = []
    for i in range(batch_size):
        fx = flow[i, 0]
        fy = flow[i, 1]
        mag = torch.sqrt(fx**2 + fy**2)
        mag = (mag - mag.min()) / (mag.max() - mag.min())
        magnitude_list.append(mag.unsqueeze(0))

    flow_magnitude = torch.cat(magnitude_list, dim=0)
    return flow_magnitude


def get_dataset_json(data_list, size=None, transform=None):
    """
    Reads data from a list of numpy arrays.
    """
    return CustomDatasetJson(data_list, size=size, transform=transform), True

def get_dataset_npy(data_list, seg_list=None):
    """
    Reads data from a numpy file.
    """
    # Create the dataset
    #data = np.load(args.images_path, mmap_mode="r")
    data = np.memmap(args.images_path, dtype='float32', mode='r', shape=(56, 625, 625, 200))


    # Split the dataset into training, validation, and test sets
    if  args.seg_path.lower() != 'none':
        #seg_data = np.load(args.seg_path)
        seg_data = np.memmap(args.seg_path, dtype='float32', mode='r', shape=(56, 625, 625, 200))
        dataset = CustomDataset(data, seg_data, transform=None)
        auxiliary_data = True

    else:
        dataset = CustomDataset(data, transform=None)
        auxiliary_data = False

    return dataset, auxiliary_data

if __name__=="__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)

    # Construct the path to the config file
    config_path = os.path.join(project_root, 'config.ini')
    
    config = configparser.ConfigParser() 
    config.read(config_path)

    training_params = config['pytorch'] # Get the training parameters

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--grid_size_x', type=int, default=training_params['grid_size_x'], help='Grid size x for the spatial transformation')
    parser.add_argument('--grid_size_y', type=int, default=training_params['grid_size_y'], help='Grid size y for the spatial transformation')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Loss Func')
    parser.add_argument('--loss_weight', type=float, default=training_params['loss_weight'], help='Smoothness Loss Weight')
    parser.add_argument('--images_path', type=str, default=training_params['images_path'], help='Path to json file containing the MRI scans as numpy array')
    parser.add_argument('--seg_path', type=str, default=training_params['seg_path'], help='Path to npy file containing the segmentation masks as numpy array')
    parser.add_argument('--weights_path', type=str, default=training_params['weights_path'], help='Path to save model weights')
    parser.add_argument('--output_dir', type=str, default=training_params['output_dir'], help='Path to save the output files')

    args = parser.parse_args()

    # Set the device
    print("Checking for CUDA availability...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("CUDA device names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("CUDA visible devices:", cuda_visible_devices())
    device = torch.device("cuda" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Set up logging
    logging.basicConfig(filename=log_dir+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Log the parameters
    logging.info(f'Data Path: {args.images_path}')
    logging.info(f'Seg Path: {args.seg_path}')
    print("Get Dataset...")
    dataset = CustomDatasetJson(args.images_path, size=(args.grid_size_y, args.grid_size_x), with_depth=True)
    auxiliary_data = True

    print("Dataset loaded. Length:", len(dataset))
    test_loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    # Create the model
    print("Creating model...")
    model = VoxelMorph(grid_size=[args.grid_size_x, args.grid_size_y], auxiliary_data=auxiliary_data)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))  # Load the model weights
    model.to(device)

    print("Testing model...")
    metrics = ImageMetrics()
    test(model, test_loader, args.loss, args.loss_weight, metrics, device, output_dir=args.output_dir, grid=[args.grid_size_x, args.grid_size_y], seg=auxiliary_data)


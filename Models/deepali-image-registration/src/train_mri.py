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

def train(model, train_loader, val_loader, optimizer, loss_func, smoothness_weight, num_epochs, device, weights_dir, output_dir, grid, seg=False):
    '''
    Train the model

    :param model: Model to train
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param optimizer: Optimizer to use
    :param loss_func: Loss function to use
    :param smoothness_weight: Weight for the smoothness loss
    :param num_epochs: Number of epochs to train
    :param device: Device to use for training

    :return: None
    '''

    if loss_func == 'MSE':
        criterion_data = VoxelMorphDataLoss(use_mse=True, smoothness_weight=smoothness_weight)
    else:
        criterion_data = VoxelMorphDataLoss(use_mse=False, smoothness_weight=smoothness_weight)
    
    if seg:
        criterion_seg = VoxelMorphSegLoss()
    else:
        criterion_seg = None
    
    criterion = VoxelMorphLoss(criterion_data, criterion_seg, seg_weight=0.01)
    best_model_filename = None
    lowest_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
        model.train()
        epoch_train_loss = 0
        for batch_idx, data in enumerate(train_loader): # Iterate over the training data
            print(f"Training batch {batch_idx + 1}/{len(train_loader)}")
            print(f"Sample shape: {len(data)}, Data 0 shape: {data[0].shape}")

            optimizer.zero_grad() # Zero the gradients

            if seg:
                source, target, source_seg, target_seg = data
                source, target, source_seg, target_seg = source.to(device).float(), target.to(device).float(), source_seg.to(device).float(), target_seg.to(device).float()
                transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass
                loss = criterion(target, transformed, flow, target_seg, transformed_seg) # Compute the loss
                print(f"Shape: Transformed: {transformed.shape}, Flow: {flow.shape}, Transformed Seg: {transformed_seg.shape}, loss: {loss.shape}")

            else:
                source, target = data
                source, target = source.to(device).float(), target.to(device).float()
                transformed, flow = model(source, target) # Forward pass
                loss = criterion(target, transformed, flow) # Compute the loss

            loss.backward() # Backward pass
            optimizer.step() # Update the weights

            epoch_train_loss += loss.item()
            
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx) # Log the loss
        
        avg_train_loss_message = f'====> Epoch: {epoch + 1} Average training loss: {epoch_train_loss / len(train_loader):.6f}'
        logging.info(avg_train_loss_message) # Log the average training loss
        
        model.eval()
        val_loss = 0
        atlas = []
        warped = []
        moving = []
        if seg:
            atlas_segs = []
            warped_segs = []
            moving_segs = []
        flows = []

        with torch.no_grad(): # Disable gradient computation for validation
            for batch_idx, data in enumerate(val_loader):
                print(f"Validating batch {batch_idx + 1}/{len(val_loader)}")
                print(f"Sample shape: {len(data)}, Data 0 shape: {data[0].shape}")

                if seg:
                    source, target, source_seg, target_seg = data
                    source, target, source_seg, target_seg = source.to(device).float(), target.to(device).float(), source_seg.to(device).float(), target_seg.to(device).float()
                    transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass
                    v_loss = criterion(target, transformed, flow, target_seg, transformed_seg) # Compute the loss

                    atlas.append(source.cpu().numpy())
                    warped.append(transformed.cpu().numpy())
                    moving.append(target_seg.cpu().numpy())
                    warped_segs.append(transformed_seg.cpu().numpy())
                    moving_segs.append(target_seg.cpu().numpy())
                    atlas_segs.append(source_seg.cpu().numpy())
                    flows.append(flow.unsqueeze(1).cpu().numpy()) # Visualize the flow magnitude
                else:
                    source, target = data
                    source, target = source.to(device).float(), target.to(device).float()
                    transformed, flow = model(source, target) # Forward pass
                    v_loss = criterion(target, transformed, flow) # Compute the loss

                    atlas.append(source.cpu().numpy())
                    warped.append(transformed.cpu().numpy())
                    flows.append(flow.unsqueeze(1).cpu().numpy()) # Visualize the flow magnitude

                val_loss += v_loss.item()
                writer.add_scalar('Loss/validation', v_loss.item(), epoch * len(val_loader) + batch_idx) # Log the loss

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss

            current_model_filename = f"best_model_epoch_{epoch + 1}"
            torch.save(model.state_dict(), f"{weights_dir}/{current_model_filename}.pth") # Save the best model
            # delete the previous best model. Name is saved in the same way
            if best_model_filename is not None:
                print(f"Removing previous best model: {best_model_filename}.pth")
                os.remove(f"{weights_dir}/{best_model_filename}.pth")
                os.remove(f"{output_dir}/{best_model_filename}_atlas.npy")
                os.remove(f"{output_dir}/{best_model_filename}_warped.npy")
                os.remove(f"{output_dir}/{best_model_filename}_moving.npy")
                if seg:
                    os.remove(f"{output_dir}/{best_model_filename}_warped_segs.npy")
                    os.remove(f"{output_dir}/{best_model_filename}_moving_segs.npy")
                    os.remove(f"{output_dir}/{best_model_filename}_atlas_segs.npy")
                os.remove(f"{output_dir}/{best_model_filename}_flow.npy")
                print(f"Removed previous best model: {best_model_filename}.pth")

            best_model_filename = current_model_filename
            print(f"Saved new best model: {best_model_filename}.pth")

            # Flatten the lists for saving
            atlas = np.concatenate(atlas, axis=0)
            warped = np.concatenate(warped, axis=0)
            moving = np.concatenate(moving, axis=0)
            print(f"Flows shape before concatenation: {len(flows)}")
            for f in flows:
                print(f"Flow shape: {f.shape}")
            flows = np.concatenate(flows, axis=0)
            if seg:
                atlas_segs = np.concatenate(atlas_segs, axis=0)
                warped_segs = np.concatenate(warped_segs, axis=0)
                moving_segs = np.concatenate(moving_segs, axis=0)
            flows = np.concatenate(flows, axis=0)

            print(f"Atlas shape: {atlas.shape}, Warped shape: {warped.shape}, Moving shape: {moving.shape}, Flows shape: {flows.shape}")
            atlas = atlas.reshape(len(atlas), 1, 1, grid[0], grid[1])
            warped = warped.reshape(len(warped), 1, 1, grid[0], grid[1])
            moving = moving.reshape(len(moving), 1, 1, grid[0], grid[1])
            if seg:
                atlas_segs = atlas_segs.reshape(len(atlas_segs), 1, 1, grid[0], grid[1])
                warped_segs = warped_segs.reshape(len(warped_segs), 1, 1, grid[0], grid[1])
                moving_segs = moving_segs.reshape(len(moving_segs), 1, 1, grid[0], grid[1])
            flows = np.array(flows).reshape(len(flows), 2, 1, grid[0], grid[1])  # Assuming flow is 2D


            np.save(os.path.join(output_dir, f"{current_model_filename}_atlas.npy"), atlas) # Save the atlas
            np.save(os.path.join(output_dir, f"{current_model_filename}_warped.npy"), warped) # Save the warped images
            np.save(os.path.join(output_dir, f"{current_model_filename}_moving.npy"), moving) # Save the moving images
            if seg:
                np.save(os.path.join(output_dir, f"{current_model_filename}_warped_segs.npy"), warped_segs) # Save the transformed segmentation masks
                np.save(os.path.join(output_dir, f"{current_model_filename}_moving_segs.npy"), moving_segs) # Save the target segmentation masks
                np.save(os.path.join(output_dir, f"{current_model_filename}_atlas_segs.npy"), atlas_segs) # Save the atlas segmentation masks
            np.save(os.path.join(output_dir, f"{current_model_filename}_flow.npy"), flows) # Save the flow magnitudes
            logging.info(f'Saved best model at epoch {epoch + 1} with validation loss {avg_val_loss:.6f}') # Log the best model
            writer.add_scalar('Loss/best_validation_loss', avg_val_loss, epoch) # Log the best validation loss

        if epoch > 0:
            os.remove(f"{weights_dir}/model_epoch_{epoch}.pth") # Remove the previous best model
        torch.save(model.state_dict(), f"{weights_dir}/model_epoch_{epoch + 1}.pth") # Save the model for the current epoch


        logging.info(f'====> Validation Epoch: {epoch + 1} Average validation loss: {avg_val_loss:.6f}') # Log the average validation loss


def test(model, test_loader, loss_func, loss_weight, metrics, device, output_dir, grid, seg=False):

    '''
    Test the model

    :param model: Model to test
    :param test_loader: DataLoader for test data
    :param loss_func: Loss function to use
    :param loss_weight: Weight for the smoothness loss
    :param metrics: Metrics object to use
    :param device: Device to use for testing

    :return: None
    '''

    model.eval()
    total_dice = 0
    total_mi = 0
    global_step = 0

    save_dir = "evaluation_outputs"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad(): # Disable gradient computation for testing
        # tqdm is used to show a progress bar for the testing loop

        atlas = []
        warped = []
        moving = []
        if seg:
            atlas_segs = []
            warped_segs = []
            moving_segs = []
        flows = []

        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', unit='batch'):
            print(f"Testing batch {batch_idx + 1}/{len(test_loader)}")
            print(f"Sample shape: {len(data)}, Data 0 shape: {data[0].shape}")
            if seg:
                source, target, source_seg, target_seg = data
                source, target, source_seg, target_seg = source.to(device), target.to(device), source_seg.to(device), target_seg.to(device)
                transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass

                print(f"Shape: Transformed: {transformed.shape}, Flow: {flow.shape}, Transformed Seg: {transformed_seg.shape}")
            else:
                source, target = data
                source, target = source.to(device), target.to(device)
                transformed, flow = model(source, target) # Forward pass

            dice_score_batch = metrics.dice_loss(transformed, target) # Compute the Dice score
            mi_score_batch = metrics.mi_loss(transformed, target) # Compute the Mutual Information score
            total_mi += mi_score_batch
            total_dice += dice_score_batch

            # 🧪 Speichern aller Outputs für jedes Sample im Batch aber nur in ein npy file
            for i in range(source.shape[0]):
                base_name = f"sample_{batch_idx}_{i}"

                # Konvertiere zu CPU + numpy
                s = source[i].squeeze().detach().cpu().numpy()
                t = target[i].squeeze().detach().cpu().numpy()
                tr = transformed[i].squeeze().detach().cpu().numpy()
                fl = flow[i].unsqueeze(1).detach().cpu().numpy()

                atlas.append(s)
                warped.append(tr)
                moving.append(t)
                flows.append(fl) # Visualize the flow magnitude
            

                if seg:
                    ss = source_seg[i].squeeze().detach().cpu().numpy()
                    ts = target_seg[i].squeeze().detach().cpu().numpy()
                    trs = transformed_seg[i].squeeze().detach().cpu().numpy()

                    atlas_segs.append(ss)
                    warped_segs.append(trs)
                    moving_segs.append(ts)

                # 📊 Logge die Metriken
                with open(os.path.join(save_dir, "metrics_log.csv"), "a") as f:
                    f.write(f"{base_name},{dice_score_batch:.6f},{mi_score_batch:.6f}\n")

            if batch_idx % 3 == 0: # Log the results for every 3rd batch
                combined = []
                for s, tr, ta in zip(source, transformed, target):
                    combined.extend([s, tr, ta])  # Add source, transformed, target, and flow magnitude to the list
                
                # Check if the images are grayscale
                if combined[0].shape[0] == 1: 
                    combined = [img.repeat(3, 1, 1) for img in combined]
                
                combined_grid = make_grid(combined, nrow=3, normalize=False, padding=5) # Create a grid of images
                
                writer.add_image('Test/Source_Transformed_Target', combined_grid, global_step) # Log the grid of images
                
                global_step += 1

        atlas = np.array(atlas).reshape(len(atlas), 1, 1, grid[0], grid[1])
        warped = np.array(warped).reshape(len(warped), 1, 1, grid[0], grid[1])
        moving = np.array(moving).reshape(len(moving), 1, 1, grid[0], grid[1])
        if seg:
            atlas_segs = np.array(atlas_segs).reshape(len(atlas_segs), 1, 1, grid[0], grid[1])
            warped_segs = np.array(warped_segs).reshape(len(warped_segs), 1, 1, grid[0], grid[1])
            moving_segs = np.array(moving_segs).reshape(len(moving_segs), 1, 1, grid[0], grid[1])
        flows = np.array(flows).reshape(len(flows), 2, 1, grid[0], grid[1])  # Assuming flow is 2D

        # Save the results as numpy arrays
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_atlas.npy"), atlas)
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_warped.npy"), warped)
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_moving.npy"), moving)
        if seg:
            np.save(os.path.join(output_dir, "voxelmorph_deepali_test_atlas_seg.npy"), atlas_segs)
            np.save(os.path.join(output_dir, "voxelmorph_deepali_test_warped_seg.npy"), warped_segs)
            np.save(os.path.join(output_dir, "voxelmorph_deepali_test_moving_segs.npy"), moving_segs)
        np.save(os.path.join(output_dir, "voxelmorph_deepali_test_flow.npy"), flows)

    avg_dice = total_dice / len(test_loader)
    avg_mi = total_mi / len(test_loader)

    logging.info(f'Average Dice Score: {avg_dice:.6f}')
    logging.info(f'Average Mutual Information Score: {avg_mi:.6f}')

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
    parser.add_argument('--batch_size', type=int, default=training_params['batch_size'], help='Batch size for data generators')
    parser.add_argument('--train_val_split', type=float, default=training_params['train_val_split'], help='First test size for splitting data')
    parser.add_argument('--val_test_split', type=float, default=training_params['val_test_split'], help='Second test size for splitting data')
    parser.add_argument('--nb_epochs', type=int, default=training_params['nb_epochs'], help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=training_params['learning_rate'], help='Learning rate (default: 0.0001)')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Loss Func')
    parser.add_argument('--loss_weight', type=float, default=training_params['loss_weight'], help='Smoothness Loss Weight')
    parser.add_argument('--images_path', type=str, default=training_params['images_path'], help='Path to npy file containing the MRI scans as numpy array')
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
    logging.info(f'Epoch: {args.nb_epochs}')
    logging.info(f'Learning Rate: {args.learning_rate}')
    logging.info(f'Loss: {args.loss}')
    logging.info(f'Loss Weight: {args.loss_weight}')


    print("Get Dataset...")
    dataset, auxiliary_data = get_dataset_json(args.images_path, size=(args.grid_size_y, args.grid_size_x)) if args.images_path.endswith('.json') else get_dataset_npy(args.images_path, args.seg_path if args.seg_path.lower() != 'none' else None)

    print("Dataset loaded. Length:", len(dataset))
    print("Splitting dataset into train, validation, and test sets...")
    # x_train, x_other = train_test_split(dataset, test_size=args.train_val_split, random_state=42)
    # x_test, x_val = train_test_split(x_other, test_size=args.val_test_split, random_state=42)

    # print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # # Create the DataLoaders
    # print("Creating DataLoaders...")
    # train_loader = DataLoader(x_train, batch_size=args.batch_size, shuffle=True, num_workers=10)
    # val_loader = DataLoader(x_val, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # test_loader = DataLoader(x_test, batch_size=1, shuffle=False, num_workers=10)

    indices = list(range(len(dataset)))
    train_frac = 1 - args.train_val_split
    val_frac   = args.train_val_split * args.val_test_split
    test_frac  = args.train_val_split * (1 - args.val_test_split)
    train_indices, val_indices, test_indices = custom_split(indices, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=42)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    # Create the model
    print("Creating model...")
    model = VoxelMorph(grid_size=[args.grid_size_x, args.grid_size_y], auxiliary_data=auxiliary_data)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Create the optimizer

    # Train the model
    print("Starting training...")
    logging.info('Training model...')
    train(model, train_loader, val_loader, optimizer, args.loss, args.loss_weight, num_epochs=args.nb_epochs, device=device, weights_dir=args.weights_path, output_dir=args.output_dir, grid=[args.grid_size_x, args.grid_size_y], seg=auxiliary_data)

    # Save the model
    print("Saving model...")
    logging.info('Saving model')
    torch.save(model.state_dict(), f"{args.weights_path}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.pth')
    logging.info(f'Model saved to {args.weights_path}')
    # Test the model
    print("Testing model...")
    metrics = ImageMetrics()
    test(model, test_loader, args.loss, args.loss_weight, metrics, device, output_dir=args.output_dir, grid=[args.grid_size_x, args.grid_size_y], seg=auxiliary_data)


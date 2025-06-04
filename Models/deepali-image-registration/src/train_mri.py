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
from classes.own_dataset import OwnDataset
from classes.losses import VoxelMorphLoss, VoxelMorphDataLoss, VoxelMorphSegLoss


log_dir = "deepali_vxl/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # TensorBoard log directory
writer = SummaryWriter(log_dir=log_dir) # Create a SummaryWriter object



def train(model, train_loader, val_loader, optimizer, loss_func, smoothness_weight, num_epochs, device, seg=False):
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

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_idx, data in enumerate(train_loader): # Iterate over the training data
            optimizer.zero_grad() # Zero the gradients

            if seg:
                source, target, source_seg, target_seg = data
                source, target, source_seg, target_seg = source.to(device).float(), target.to(device).float(), source_seg.to(device).float(), target_seg.to(device).float()
                transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass
                loss = criterion(target, transformed, flow, target_seg, transformed_seg) # Compute the loss
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

        with torch.no_grad(): # Disable gradient computation for validation
            for batch_idx, data in enumerate(val_loader):
                if seg:
                    source, target, source_seg, target_seg = data
                    source, target, source_seg, target_seg = source.to(device).float(), target.to(device).float(), source_seg.to(device).float(), target_seg.to(device).float()
                    transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass
                    v_loss = criterion(target, transformed, flow, target_seg, transformed_seg) # Compute the loss
                else:
                    source, target = data
                    source, target = source.to(device).float(), target.to(device).float()
                    transformed, flow = model(source, target) # Forward pass
                    v_loss = criterion(target, transformed, flow) # Compute the loss

                val_loss += v_loss.item()
                writer.add_scalar('Loss/validation', v_loss.item(), epoch * len(val_loader) + batch_idx) # Log the loss

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'====> Validation Epoch: {epoch + 1} Average validation loss: {avg_val_loss:.6f}') # Log the average validation loss


def test(model, test_loader, loss_func, loss_weight, metrics, device, seg=False):

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

    with torch.no_grad(): # Disable gradient computation for testing
        for batch_idx, data in enumerate(test_loader): # Iterate over the test data
            
            if seg: 
                source, target, source_seg, target_seg = data
                source, target, source_seg, target_seg = source.to(device), target.to(device), source_seg.to(device), target_seg.to(device)
                transformed, flow, transformed_seg = model((source, source_seg), target) # Forward pass
            else:
                source, target = data
                source, target = source.to(device), target.to(device)
                transformed, flow = model(source, target) # Forward pass

            dice_score_batch = metrics.dice_loss(transformed, target) # Compute the Dice score
            mi_score_batch = metrics.mi_loss(transformed, target) # Compute the Mutual Information score
            total_mi += mi_score_batch
            total_dice += dice_score_batch

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

    avg_dice = total_dice / len(test_loader)
    avg_mi = total_mi / len(test_loader)

    logging.info(f'Average Dice Score: {avg_dice:.6f}')
    logging.info(f'Average Mutual Information Score: {avg_mi:.6f}')

def visualize_flow(flow):
    '''
    Get the flow magnitude for visualization

    :param flow: Flow tensor

    :return: Flow magnitude tensor
    '''
    flow_x = flow[0, 0, :, :].unsqueeze(0)  # X displacement
    flow_y = flow[0, 1, :, :].unsqueeze(0)  # Y displacement

    # Normalize to range [0, 1] for visualization
    flow_x = (flow_x - flow_x.min()) / (flow_x.max() - flow_x.min())
    flow_y = (flow_y - flow_y.min()) / (flow_y.max() - flow_y.min())

    # Optionally: Compute flow magnitude and log it as well
    flow_magnitude = torch.sqrt(flow_x ** 2 + flow_y ** 2)
    flow_magnitude = (flow_magnitude - flow_magnitude.min()) / (flow_magnitude.max() - flow_magnitude.min())

    return flow_magnitude

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

    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")
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

    # Create the dataset
    data = np.load(args.images_path)

    # Split the dataset into training, validation, and test sets
    if  args.seg_path.lower() != 'none':
        seg_data = np.load(args.seg_path)
        dataset = CustomDataset(data, seg_data, transform=None)
        auxiliary_data = True

    else:
        dataset = CustomDataset(data, transform=None)
        auxiliary_data = False

    x_train, x_other = train_test_split(dataset, test_size=args.train_val_split, random_state=42)
    x_test, x_val = train_test_split(x_other, test_size=args.val_test_split, random_state=42)

    train_loader = DataLoader(x_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(x_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(x_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create the model
    model = VoxelMorph(grid_size=[args.grid_size_x, args.grid_size_y], auxiliary_data=auxiliary_data)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Create the optimizer

    # Train the model
    logging.info('Training model...')
    train(model, train_loader, val_loader, optimizer, args.loss, args.loss_weight, num_epochs=args.nb_epochs, device=device, seg=auxiliary_data)

    # Save the model
    logging.info('Saving model')
    torch.save(model.state_dict(), args.weights_path)

    # Test the model
    metrics = ImageMetrics()
    test(model, test_loader, args.loss, args.loss_weight, metrics, device, seg=auxiliary_data)

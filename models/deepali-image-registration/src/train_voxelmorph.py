import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # Required for VoxelMorph to work with TensorFlow 2.x

import argparse
import datetime

import voxelmorph as vxm
from voxelmorph.tf.networks import VxmDense

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import configparser

import sys
import os

import logging

import torch
from deepali.core.environ import cuda_visible_devices

from classes.own_dataset_json import CustomDataset as CustomDatasetJson
from torch.utils.data import DataLoader
import random


sys.path.append(os.getcwd())

log_dir = "vxlmorph/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # TensorBoard log directory
os.makedirs(log_dir, exist_ok=True) # Create the log directory


def vxm_data_generator(paired_data, batch_size=16):
    """
    Generator that takes in pre-paired data of size [N, H, W], and yields data for
    our custom vxm model.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]

    paired_data: numpy array of shape [N, 2, H, W], where paired_data[:, 0] 
    is the moving images and paired_data[:, 1] is the fixed images.
    """
    vol_shape = paired_data.shape[2:-1]  # Shape of the volume [H, W]
    ndims = len(vol_shape)  # Number of dimensions (e.g., 2 for 2D images)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])  # Zero-gradient field
    
    while True:
        # Randomly select indices for the batch
        idx = np.random.randint(0, paired_data.shape[0], size=batch_size)
        
        # Extract moving and fixed images from the paired data
        moving_images = paired_data[idx, 0, ...]  # [bs, H, W, 1]
        fixed_images = paired_data[idx, 1, ...]   # [bs, H, W, 1]
        
        # Inputs for the model
        inputs = [moving_images, fixed_images]
        
        # Outputs for the model
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

def torch_to_keras_generator(torch_loader):
    import numpy as np
    while True:
        for atlas_img, moving_img, atlas_seg, moving_seg in torch_loader:
            atlas_img_np = atlas_img.numpy().transpose(0, 2, 3, 1)
            moving_img_np = moving_img.numpy().transpose(0, 2, 3, 1)
            vol_shape = atlas_img_np.shape[1:-1]
            ndims = len(vol_shape)
            zero_phi = np.zeros([atlas_img_np.shape[0], *vol_shape, ndims])
            inputs = [moving_img_np, atlas_img_np]
            outputs = [atlas_img_np, zero_phi]
            yield (inputs, outputs)

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

def combine_images(moving, fixed, moved):
    # Assuming images are [batch_size, height, width, channels]
    return tf.concat([moving, fixed, moved], axis=2)


if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)

    # Construct the path to the config file
    config_path = os.path.join(project_root, 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_path)

    training_params = config['tensorflow']

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--batch_size', type=int, default=training_params['batch_size'], help='Batch size for data generators')
    parser.add_argument('--train_val_split', type=float, default=training_params['train_val_split'], help='First test size for splitting data')
    parser.add_argument('--val_test_split', type=float, default=training_params['val_test_split'], help='Second test size for splitting data')
    parser.add_argument('--int_steps', type=int, default=training_params['int_steps'], help='Integration steps for VxmDense model')
    parser.add_argument('--lambda_param', type=float, default=training_params['lambda_param'], help='Lambda parameter for loss weights')
    parser.add_argument('--steps_per_epoch', type=int, default=training_params['steps_per_epoch'], help='Steps per epoch during training')
    parser.add_argument('--nb_epochs', type=int, default=training_params['nb_epochs'], help='Number of epochs for training')
    parser.add_argument('--verbose', type=int, default=training_params['verbose'], help='Verbose mode')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Type of loss function')
    parser.add_argument('--grad_norm_type', type=str, choices=['l1', 'l2'], default=training_params['grad_norm_type'], help='Type of norm for Grad loss (l1 or l2)')
    parser.add_argument('--gamma_param', type=float, default=training_params['gamma_param'], help='weight of dice loss (gamma) (default: 0.02)')
    parser.add_argument('--learning_rate', type=float, default=training_params['learning_rate'], help='Learning rate (default: 0.0001)')
    parser.add_argument('--images_path', type=str, default=training_params['images_path'], help='Path to npy file containing the MRI scans as numpy array')
    parser.add_argument('--weights_path', type=str, default=training_params['weights_path'], help='Path to save model weights')


    args = parser.parse_args()

    logging.basicConfig(filename=log_dir+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Set the device
    print("Checking for CUDA availability...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("CUDA device names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("CUDA visible devices:", cuda_visible_devices())
    device = torch.device("cuda" if torch.cuda.is_available() and cuda_visible_devices() else "cpu")
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set the device
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f" {len(physical_devices)} GPU(s) is/are available")
    else:
        logging.info("No GPU detected")
    
    # load the images
    #images = np.load(args.images_path)
    dataset = CustomDatasetJson(data_list=args.images_path, size=(512 , 512 ))  # Passe size ggf. an!
    indices = list(range(len(dataset)))

    logging.info(f'Loaded images from {args.images_path}')

    # Split the data into training, validation, and test sets
    # x_train, x_other = train_test_split(dataset, test_size=args.train_val_split, random_state=42)
    # x_test, x_val = train_test_split(x_other, test_size=args.val_test_split, random_state=42)

    # train_loader = DataLoader(x_train, batch_size=args.batch_size, shuffle=True)
    # val_loader   = DataLoader(x_val, batch_size=args.batch_size, shuffle=False)
    # test_loader  = DataLoader(x_test, batch_size=args.batch_size, shuffle=False)

    train_frac = args.train_val_split
    val_frac   = (1 - args.train_val_split) * args.val_test_split
    test_frac  = (1 - args.train_val_split) * (1 - args.val_test_split)
    train_indices, val_indices, test_indices = custom_split(indices, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=42)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)


    print(f"Training set size: {len(train_indices)}, Validation set size: {len(val_indices)}, Test set size: {len(test_indices)}")

    # Create the data generators
    train_gen = torch_to_keras_generator(train_loader)
    val_gen = torch_to_keras_generator(val_loader)
    test_gen = torch_to_keras_generator(test_loader)

    # Define the network architecture
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]

    # Create the VxmDense model
    inshape = next(train_gen)[0][0].shape[1:-1]
    logging.info(inshape)
    vxm_model = VxmDense(inshape=inshape, nb_unet_features=[enc_nf, dec_nf], int_steps=args.int_steps)
    if args.weights_path:
        vxm_model.load_weights(args.weights_path)
  # Load the model weights


    # Define the loss function
    if args.loss == 'MSE':
        loss_func = vxm.losses.MSE().loss
    elif args.loss == 'NCC':
        loss_func = vxm.losses.NCC().loss
    elif args.loss == 'MI':
        loss_func = vxm.losses.MutualInformation().loss
    elif args.loss == 'TukeyBiweight':
        loss_func = vxm.losses.TukeyBiweight().loss
    else:
        loss_func = vxm.losses.MSE().loss

    # define grad
    if args.grad_norm_type == 'l1':
        grad_norm = 'l1'
    elif args.grad_norm_type == 'l2':
        grad_norm = 'l2'
    else:
        grad_norm = 'l2'

    losses = [loss_func, vxm.losses.Grad(grad_norm).loss, vxm.losses.Dice().loss]
    loss_weights = [1, args.lambda_param, args.gamma_param]

    print(f"Loss function: {args.loss}, Grad norm type: {grad_norm}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}")

    # compile model
    logging.info('Compiling model...')
    with tf.device('/GPU:0'):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        file_writer = tf.summary.create_file_writer(log_dir)
        #file_writer = tf.summary.create_file_writer(log_dir + "/images")
        vxm_model.compile(tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=losses, loss_weights=loss_weights)
        # train and validate model
        logging.info(f'Training model with hyperparams: Loss: {args.loss}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}, Learning rate: {args.learning_rate}')
        vxm_model.fit(train_gen, steps_per_epoch=args.steps_per_epoch, epochs=args.nb_epochs, validation_data=val_gen, validation_steps=args.steps_per_epoch, verbose=args.verbose, callbacks=[tensorboard_callback])
        # save model
        logging.info('Saving model...')
        vxm_model.save_weights(args.weights_path)
        # evaluate or test model
        logging.info('Evaluating model...')
        vxm_model.evaluate(test_gen, steps=args.steps_per_epoch, verbose=args.verbose)
        # predict model and calculate the dice score between the predicted and ground truth images
        logging.info('Predicting model...')
        dice_scores = []
        mutual_info_scores = []
        global_step = 0
        test_steps = len(test_dataset)

        atlases = []
        moving_images = []
        warped_images = []
        moving_segs = []
        warped_segs = []
        atlas_segs = []
        flows = []

        for i in range(test_steps):
            print(f"Processing test step {i+1}/{test_steps}", flush=True)
            test_input, test = next(test_gen)

            print(f"Shape test_input: {len(test_input)}", flush=True)
            print(f"Shape test_input: {len(test)}", flush=True)

            test_pred = vxm_model.predict(test_input, verbose=args.verbose)

            print(f"Shape test_pred: {len(test_pred)}, Test_pred: {test_pred[0].shape} , {test_pred[1].shape}", flush=True)
            print(f"Shape test_pred: {len(test)}, Test_pred: {test[0].shape} , {test[1].shape}", flush=True)

            test[0] = np.transpose(test[0], (0, 3, 1, 2))
            test[1] = np.transpose(test[1], (0, 3, 1, 2))

            if i % 1 == 0:
                with file_writer.as_default():
                    # Combine the images
                    combined_image = combine_images(test_input[1], test_input[0], test_pred[0])

                    # Write the combined image to TensorBoard
                    tf.summary.image(f"Combined_Image", combined_image, step=global_step, max_outputs=args.batch_size)

            atlases.append(test_input[0])
            moving_images.append(test_input[1])
            warped_images.append(test_pred[0])

            atlas_segs.append(test[0])
            warped_segs.append(test[1])

            print(f"Shape test_input: {len(test_input)}, Shape test_pred: {len(test_pred)}", flush=True)

            test_input = tf.convert_to_tensor(test_input[1], dtype=tf.float32)
            test_pred = tf.convert_to_tensor(test_pred[0], dtype=tf.float32)
            dice = vxm.losses.Dice().loss(tf.cast((test_input >= 0.5), dtype=float), tf.cast((test_pred >= 0.5), dtype=float))
            mi = vxm.losses.MutualInformation().loss(test_input, test_pred)
            dice_scores.append(dice)
            mutual_info_scores.append(mi)
            global_step += 1

        # Convert lists to numpy arrays
        atlases = np.array(atlases)
        moving_images = np.array(moving_images)
        warped_images = np.array(warped_images)
        atlas_segs = np.array(atlas_segs) if atlas_segs else None
        warped_segs = np.array(warped_segs) if warped_segs else None
        flows = np.array(flows) if flows else None

        # Save the results
        np.save(os.path.join(log_dir, 'atlases.npy'), atlases)
        np.save(os.path.join(log_dir, 'moving_images.npy'), moving_images)
        np.save(os.path.join(log_dir, 'warped_images.npy'), warped_images)
        if atlas_segs is not None:
            np.save(os.path.join(log_dir, 'atlas_segs.npy'), atlas_segs)
        if warped_segs is not None:
            np.save(os.path.join(log_dir, 'warped_segs.npy'), warped_segs)
        if flows is not None:
            np.save(os.path.join(log_dir, 'flows.npy'), flows)

        average_dice_score = np.mean(dice_scores)
        average_mutual_info_score = np.mean(mutual_info_scores)
        logging.info(f'Average dice score: {average_dice_score}')
        logging.info(f'Average mutual information score: {average_mutual_info_score}')

        logging.info(f'Model hyperparams: Loss: {args.loss}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}, Learning rate: {args.learning_rate} - Average dice score: {average_dice_score}')
        #np.save(f'vxlmorph/tensorboard/Semisupervised/Metrics/Dice_hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}.npy', np.array(dice_scores))
        
        logging.info('\n---------------------------------------------------------------------------------------------------------\n')
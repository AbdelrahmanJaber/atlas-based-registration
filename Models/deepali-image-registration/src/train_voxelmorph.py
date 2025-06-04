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

sys.path.append(os.getcwd())

log_dir = "vxlmorph/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # TensorBoard log directory
os.mkdir(log_dir) # Create the log directory


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
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f" {len(physical_devices)} GPU(s) is/are available")
    else:
        logging.info("No GPU detected")
    
    # load the images
    images = np.load(args.images_path)

    logging.info(f'Loaded images from {args.images_path}')

    # Split the data into training, validation, and test sets
    x_train, x_other = train_test_split(images, test_size=args.train_val_split, random_state=42)
    x_test, x_val = train_test_split(x_other, test_size=args.val_test_split, random_state=42)

    # Create the data generators
    train_gen = vxm_data_generator(x_train, batch_size=args.batch_size)
    val_gen = vxm_data_generator(x_val, batch_size=args.batch_size)
    test_gen = vxm_data_generator(x_test, batch_size=args.batch_size)

    # Define the network architecture
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]

    # Create the VxmDense model
    inshape = next(train_gen)[0][0].shape[1:-1]
    logging.info(inshape)
    vxm_model = VxmDense(inshape=inshape, nb_unet_features=[enc_nf, dec_nf], int_steps=args.int_steps)

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
        test_steps = len(x_test) // args.batch_size
        for i in range(test_steps):
            test_input, _ = next(test_gen)

            test_pred = vxm_model.predict(test_input, verbose=args.verbose)

            if i % 3 == 0:
                with file_writer.as_default():
                    # Combine the images
                    combined_image = combine_images(test_input[0], test_input[1], test_pred[0])
                    
                    # Write the combined image to TensorBoard
                    tf.summary.image(f"Combined_Image", combined_image, step=global_step, max_outputs=args.batch_size)

            
            test_input = tf.convert_to_tensor(test_input[1], dtype=tf.float32)
            test_pred = tf.convert_to_tensor(test_pred[0], dtype=tf.float32)
            dice = vxm.losses.Dice().loss(tf.cast((test_input >= 0.5), dtype=float), tf.cast((test_pred >= 0.5), dtype=float))
            mi = vxm.losses.MutualInformation().loss(test_input, test_pred)
            dice_scores.append(dice)
            mutual_info_scores.append(mi)
            global_step += 1

        average_dice_score = np.mean(dice_scores)
        average_mutual_info_score = np.mean(mutual_info_scores)
        logging.info(f'Average dice score: {average_dice_score}')
        logging.info(f'Average mutual information score: {average_mutual_info_score}')

        logging.info(f'Model hyperparams: Loss: {args.loss}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}, Learning rate: {args.learning_rate} - Average dice score: {average_dice_score}')
        #np.save(f'vxlmorph/tensorboard/Semisupervised/Metrics/Dice_hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}.npy', np.array(dice_scores))
        
        logging.info('\n---------------------------------------------------------------------------------------------------------\n')
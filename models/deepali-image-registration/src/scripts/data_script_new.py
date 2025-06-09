import numpy as np
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
import random

def load_mri_data(file_path):
    """Load MRI data from a NumPy file."""
    return np.load(file_path)

def preprocess_mri(image):
    """Apply preprocessing steps to an MRI image."""
    # Normalize pixel values to [-1, 1] range
    scaler = MinMaxScaler()
    image_normalized = scaler.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
    
    # Enhance contrast using adaptive histogram equalization
    image_equalized = exposure.equalize_adapthist(image_normalized)
    
    return image_equalized

def generate_random_pairs(images):
    """Generate random pairs of images, ensuring each image is a target exactly once."""
    num_images = len(images)
    pairs = []
    
    for i in range(num_images):
        target = images[i]
        source_index = i
        while source_index == i:
            source_index = random.randint(0, num_images - 1)
        source = images[source_index]
        pairs.append((target, source))
    
    return pairs

def main():
    # Load MRI data
    input_file_path = 'data/brats_flair_slices.npy'
    mri_data = load_mri_data(input_file_path)
    
    # Preprocess all images
    preprocessed_images = [preprocess_mri(img) for img in mri_data]
    
    # Generate random pairs
    image_pairs = generate_random_pairs(preprocessed_images)
    num_pairs = len(image_pairs)
    
    # Prepare the array for saving
    x, y = preprocessed_images[0].shape
    pairs_array = np.zeros((num_pairs, 2, x, y, 1), dtype=np.float32)
    
    for i, (img1, img2) in enumerate(image_pairs):
        pairs_array[i, 0, :, :, 0] = img1
        pairs_array[i, 1, :, :, 0] = img2
    
    # Save the pairs to a single .npy file
    output_file_path = 'data/brats_flair_dataset.npy'
    np.save(output_file_path, pairs_array)
    
    print(f"Generated and saved {num_pairs} random pairs of preprocessed MRI images.")
    print(f"Output shape: {pairs_array.shape}")

if __name__ == "__main__":
    main()
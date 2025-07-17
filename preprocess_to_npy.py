#ich will die pairs aus einer json lesen und dann die Bilder preprocessen und speichern
import numpy as np
import json
from tqdm import tqdm
from TPTBox import NII
import numpy as np
import argparse
import torch

def center_crop(array, target_shape):
    """
    Cropt das Array zentriert auf target_shape (z. B. (64, 64, 64)).
    """
    current_shape = array.shape
    slices = []

    for curr, tgt in zip(current_shape, target_shape):
        start = max((curr - tgt) // 2, 0)
        end = start + tgt
        slices.append(slice(start, end))

    return array[tuple(slices)]

def pad_or_crop_center(array, target_shape, pad_value=0):
    shape = array.shape
    diffs = [t - s for s, t in zip(shape, target_shape)]

    # Padding falls nötig
    pad_width = [(max(d // 2, 0), max(d - d // 2, 0)) for d in diffs]
    array = np.pad(array, pad_width, mode='constant', constant_values=pad_value)

    # Cropping falls nötig
    start = [(array.shape[i] - target_shape[i]) // 2 for i in range(len(target_shape))]
    slices = tuple(slice(s, s + target_shape[i]) for i, s in enumerate(start))
    return array[slices]

def preprocess_pair(pair, target_shape, crop_shape=None):
    raw_nii = NII.load(pair[0], seg=False)
    seg_nii = NII.load(pair[1], seg=True)

    #raw_nii = raw_nii.reorient(axcodes_to=("L", "A", "S"))
    raw_nii = raw_nii.rescale(target_shape)

    #seg_nii = seg_nii.reorient(axcodes_to=("L", "A", "S"))
    #seg_nii = raw_nii.resample_from_to(seg_nii)

    raw_array = raw_nii.get_array()
    seg_array = seg_nii.get_array()

    if crop_shape is None:
        crop_shape = raw_array.shape #croppen oder padden je nach dem was kleiner ist
    
    # if raw_array.shape < crop_shape:
    #     padding = [(0, max(0, cs - rs)) for cs, rs in zip(crop_shape, raw_array.shape)]
    #     raw_array = np.pad(raw_array, padding, mode='constant', constant_values=0)
    #     seg_array = np.pad(seg_array, padding, mode='constant', constant_values=0)
    # else:
    #     raw_array = center_crop(raw_array, crop_shape)
    #     seg_array = center_crop(seg_array, crop_shape)
    needs_padding = any(r < c for r, c in zip(raw_array.shape, crop_shape))
    needs_cropping = any(r > c for r, c in zip(raw_array.shape, crop_shape))

    print(f"Needs padding: {needs_padding}, Needs cropping: {needs_cropping}")

    # Padding oder Cropping
    raw_array = pad_or_crop_center(raw_array, crop_shape)
    seg_array = pad_or_crop_center(seg_array, crop_shape)

    assert raw_array.shape == crop_shape, f"Raw shape mismatch: {raw_array.shape} - crop shape: {crop_shape}"
    assert seg_array.shape == crop_shape, f"Seg shape mismatch: {seg_array.shape} - crop shape: {crop_shape}"

    return raw_array, seg_array, crop_shape


def preprocess_pair_with_rescale():
    output_raw_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/results/raw_images_6_try.npy"
    output_seg_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/results/seg_images_6_try.npy"

    with open("/vol/miltank/users/hothum/Documents/atlas-based-registration/results/paired_raw_seg.json", "r") as f:
        pairs = json.load(f)

    raw_images = []
    seg_images = []
    target_shape = (0.8, 0.8, 0.8)  # Define the target shape for rescaling
    crop_shape = None

    for index, pair in tqdm(pairs.items()):

        if index == "2":
            print("Skipping index 1 as it is not a valid pair")
            break
        # Preprocess the raw image
        raw_image, seg_image, crop_shape = preprocess_pair(pair, target_shape=target_shape, crop_shape=crop_shape)

        raw_images.append(raw_image)
        seg_images.append(seg_image)

        try:
            i = int(index)
        except ValueError:
            i = 0  # Fallback-Wert

        if i % 1 == 0:
            print(f"Processed {index} pairs")
            print(f"Intermediate results get saved to {output_raw_path} and {output_seg_path}")
            np.save(output_raw_path, np.array(raw_images))
            np.save(output_seg_path, np.array(seg_images))

    # Convert lists to numpy arrays
    raw_images_np = np.array(raw_images)
    seg_images_np = np.array(seg_images)

    # Save the numpy arrays
    np.save(output_raw_path, raw_images_np)
    np.save(output_seg_path, seg_images_np)

    print(f"Raw images saved to {output_raw_path}")
    print(f"Segmentation images saved to {output_seg_path}")


def preprocess_voxelmorph():
    file_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/Models/deepali-image-registration/data/ixi_t2_dataset_cropped.npy"
    save_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/Models/deepali-image-registration/data/ixi_t2_dataset_cropped_voxelmorph.npy"
    
    filed = np.load(file_path)
    
    print(f"Before: {filed.shape}")
    filed = torch.tensor(filed)
    filed = filed.repeat(1, 1, 1, 1, 6)
    filed = filed.numpy()
    print(f"After: {filed.shape}")

    np.save(save_path, filed)

if __name__ == "__main__":
    #KANNST DU MIR EINEN COMMANDLINE READER GEBEN; DER EIN POSITIONAL DINGS NIMMT MIT DEN  AUSWAHLMÖGLICHKEITEN NORMAL, VOXELMORPH
   parser = argparse.ArgumentParser(description="Process MRI images")
   parser.add_argument("mode", choices=["NORMAL", "VOXELMORPH"], help="Processing mode")
   args = parser.parse_args()

   if args.mode == "NORMAL":
       preprocess_pair_with_rescale()
   elif args.mode == "VOXELMORPH":
       preprocess_voxelmorph()
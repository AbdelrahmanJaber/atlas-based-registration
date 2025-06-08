#ich will die pairs aus einer json lesen und dann die Bilder preprocessen und speichern
import numpy as np
import json
from tqdm import tqdm
from TPTBox import NII
import numpy as np

def preprocess_pair(pair, target_shape):
    raw_nii = NII.load(pair[0], seg=False)
    seg_nii = NII.load(pair[1], seg=True)

    raw_nii = raw_nii.reorient(axcodes_to=("P", "I", "R"))
    raw_nii = raw_nii.rescale(target_shape)

    seg_nii = raw_nii.resample_from_to(seg_nii)

    return raw_nii.get_array(), seg_nii.get_array()

output_raw_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/results/raw_images_2_try.npy"
output_seg_path = "/vol/miltank/users/hothum/Documents/atlas-based-registration/results/seg_images_2_try.npy"

with open("/vol/miltank/users/hothum/Documents/atlas-based-registration/results/paired_raw_seg.json", "r") as f:
    pairs = json.load(f)

raw_images = []
seg_images = []
target_shape = (2, 2, 5)  # Define the target shape for rescaling

for index, pair in tqdm(pairs.items()):
    # Preprocess the raw image
    raw_image, seg_image = preprocess_pair(pair, target_shape=target_shape)
    raw_images.append(raw_image)
    seg_images.append(seg_image)

    try:
        i = int(index)
    except ValueError:
        i = 0  # Fallback-Wert

    if i % 10 == 0:
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
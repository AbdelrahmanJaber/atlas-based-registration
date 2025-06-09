import json
import nibabel as nib
import numpy as np

json_path = "/vol/miltank/users/fratzke/Documents/atlas-based-registration/results/bone_raw.json"
output_path = "/vol/miltank/users/fratzke/Documents/atlas-based-registration/results/bone_raw.npy"

json_path2 = "/vol/miltank/users/fratzke/Documents/atlas-based-registration/results/bone_seg.json"
output_path2 = "/vol/miltank/users/fratzke/Documents/atlas-based-registration/results/bone_seg.npy"

with open(json_path, 'r') as json_file:
    data_list = json.load(json_file)

# Convert the list to a NumPy array
data_array = np.array(data_list)

np.save(output_path, data_array, allow_pickle=True)
print(f"saved in: {output_path}")

with open(json_path2, 'r') as json_file:
    data_list2 = json.load(json_file)

# Convert the list to a NumPy array
data_array2 = np.array(data_list2)

np.save(output_path2, data_array2, allow_pickle=True)
print(f"saved in: {output_path2}")
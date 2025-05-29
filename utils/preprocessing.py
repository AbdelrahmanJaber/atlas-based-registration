# from pathlib import Path
# from TPTBox import BIDS_FILE
# from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_total_seg

# # Path to one test CT file
# path_to_ct = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom/rawdata/CTFU00065/ses-20100514/sub-CTFU00065_ses-20100514_sequ-2_ct.nii.gz")
# dataset_root = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom"

# # Wrap into BIDS structure
# img_bids = BIDS_FILE(path_to_ct, dataset_root)

# # Define output segmentation path
# out_file = img_bids.get_changed_path("nii.gz", "msk", parent="test", info={"seg": "bone"})

# # Run segmentation (dataset_id=520 → bone)
# if not out_file.exists():
#     run_total_seg(img_bids.get_nii_file(), out_file, dataset_id=520, padd=5, override=False)


from glob import glob
from pathlib import Path
from TPTBox import BIDS_FILE
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_total_seg

base = Path("/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/dataset-myelom")
all_ct_paths = glob(str(base / "rawdata" / "*" / "ses-*" / "*ct.nii.gz"))

for ct_path in all_ct_paths:
    ct_path = Path(ct_path)
    print(f"Processing CT file: {ct_path.name}")

    img_bids = BIDS_FILE(ct_path, base)
    out_file = img_bids.get_changed_path("nii.gz", "msk", parent="preprocessed", info={"seg": "bone"})
    
    if not out_file.exists():
        print(f"Running segmentation on: {ct_path.name}")
        run_total_seg(img_bids.get_nii_file(), out_file, dataset_id=520, padd=5, override=False)

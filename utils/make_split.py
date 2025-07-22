from models.voxelmorph.dataset import split_subjects

if __name__ == '__main__':
    split_subjects(
        data_root='data/dataset-myelom/rawdata',
        output_json='subject_splits.json'
    )
    print('subject_splits.json created successfully.') 
    
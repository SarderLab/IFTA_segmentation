### H-AI-L (Human-A.I.-Loop) for semantic segmentation of WSI (whole slide images)


This readme contains information on running IFTA segmentation on HiperGator


Clone the IFTA_segmentation repository(https://github.com/SarderLab/IFTA_segmentation) and switch to `ifta_hpg_tf2` branch

```Bash
    git clone https://github.com/SarderLab/IFTA_segmentation.git
    cd IFTA_segmentation
    git checkout ifta_hpg_tf2
```

### Steps

1. Create a folder with the project name specified in the slurm file. For example if project name is 'TxR01' create `/orange/.../IFTA_segmentation/TxR01`
    - The `TxR01/` folder will be in the same directory as where the `Codes/` folder is

2. In `/orange/.../IFTA_segmentation/TxR01` create `IFTA_segmentation/TxR01/TRAINNING_data/0` and put the whole slide images for prediction
    - This is where we store the `.svs`input image files

3. Create `/orange/.../IFTA_segmentation/TxR01/TRAINNING_data/Predited_XMLs` to save the output annotation files
    - This is where the prediction XMLs will be generated

4. Create a MODELS directory `/orange/.../IFTA_segmentation/TxR01/MODELS/0/HR` to upload the model files
    - Should have atleast the `.index` and `.data` file.

### Running in HPG

To run edit the run.sh script with new specific folder name and run:
```Bash
    sbatch run.sh
```

### H-AI-L (Human-A.I.-Loop) for semantic segmentation of WSI (whole slide images)


This readme contains information on running IFTA segmentation on HiperGator


Clone the IFTA_segmentation repository(https://github.com/SarderLab/IFTA_segmentation) and switch to sumanth_ifta_hpg branch


    git clone https://github.com/SarderLab/IFTA_segmentation.git
    cd IFTA_segmentation
    git checkout sumanth_ifta_hpg


create a folder with the project name specified in the slurm file. For example if project name is 'TxR01' create '/orange/pinaki.sarder/sdevarasetty/IFTA_segmentation/TxR01'

In '/orange/pinaki.sarder/sdevarasetty/IFTA_segmentation/TxR01' create 'IFTA_segmentation/TxR01/TRAINNING_data/0' and put the whole slide images for prediction

create '/orange/pinaki.sarder/sdevarasetty/IFTA_segmentation/TxR01/TRAINNING_data/Predited_XMLs' to save the output annotation files


Create a MODELS directory '/orange/pinaki.sarder/sdevarasetty/IFTA-Jeong-Running/H-AI-L/TxR01/MODELS/0/HR' to upload the model files


To run the code
    
    sbatch run.sh

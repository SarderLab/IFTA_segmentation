# IFTA_segmentation

This repository contains shared materials for the JASN manuscript entitled Automated Computational Detection of Interstitial Fibrosis, Tubular Atrophy, and Glomerulosclerosis, submitted to JASN in November 2020. Shared materials include example whole slide images and their CNN segmented output, and codes and pretrained CNN  model to perform IFTA and glomerulosclerosis segmentation on new whole slide images.


# Whole slide images
To view the CNN segmentationon whole slide images, you must have Aperio ImageScope (https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) installed on your computer. Then, simply place the whole slide image (.svs file) and the annotations (.xml file) in the same directory on your computer. Upon opening the .svs file with ImageScope, you will see the CNN predictions overlaid on the whole slide. We provide some example pre-generated segmentations on whole slide images here: https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e.


# Performing segmentation on new data
Before segmenting your own whole slides, you will need to:

1) Configure HAIL (https://github.com/SarderLab/H-AI-L) and its dependencies on your computer that will perform the segmentation
2) Download the pre-trained model file available at https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e 

After everything is installed, navigate to the directory where you have installed the HAIL and call the following command:


    python segmentation_school.py --option new --project your_project_name


Where --option tells HAIL to create a new directory for a new project which has name specified by --project


Navigate inside the newly created folder "your_project_name", and find the folder MODELS/0/HR/. Put the three files you downloaded from step 2) above into this folder. 

Go back up to the directory "your_project_name". Find the folder TRAINING_data/0/, and place any whole slide images for segmentation (.svs files) inside this folder. 

From a terminal prompt, while inside the directory that contains "segmentation_school.py" (i.e., where the HAIL codes are downloaded), call the following command:

    python segmentation_school.py --option predict --project your_project_name --one_network True --encoder_name deeplab --classNum 4 --boxSizeHR 3000 --overlap_percentHR 0.5

Adapt the boxSizeHR parameter to a lower value if your GPU does not have enough memory. The shared segmentation files found at the link above were morphologically post-processed to remove IFTA regions with size < 1730µm<sup>2</sup> glomerular regions with size < 1500µm<sup>2</sup>.

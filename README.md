# IFTA and glomerulosclerosis segmentation

This readme contains information on accessing the shared materials for the JASN manuscript entitled "Automated Computational Detection of Interstitial Fibrosis, Tubular Atrophy, and Glomerulosclerosis", submitted to JASN in November 2020. Shared materials include a pre-trained CNN model and corresponding codes to perform whole slide segmentation of IFTA and glomerulosclerosis on new renal biopsies. Also included are example whole slide images and their CNN segmented output.


# Whole slide images
To view the CNN segmentation on whole slide images, you must have Aperio ImageScope (https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) installed on your computer. Then, simply place the whole slide image (.svs file) and its corresponding annotations (.xml file) in the same directory on your computer. Upon opening the .svs file with ImageScope, you will see the CNN predictions overlaid on the whole slide. We provide some example pre-generated segmentations on whole slide images here: https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e. These segmentation outputs were morphologically post-processed to remove IFTA regions with size < 1730µm<sup>2</sup> glomerular regions with size < 1500µm<sup>2</sup> from the whole slide mask.



# Performing segmentation on new data
Before segmenting your own whole slides, you will need to:

1) Configure HAIL (https://github.com/SarderLab/H-AI-L) and its dependencies on your computer that will perform the segmentation
2) Download the 3 pre-trained model files available at https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e 

After dependencies are installed, navigate to the directory where you have downloaded the HAIL codes and call the following command:


    python segmentation_school.py --option new --project your_project_name


Where --option tells HAIL to create a new directory for a new project which has name specified by --project


Navigate inside the newly created folder "your_project_name", and find the folder MODELS/0/HR/. Put the three files you downloaded from step 2) above into this folder. 

Go back up to the directory "your_project_name". Find the folder TRAINING_data/0/, and place any whole slide images for segmentation (.svs files) inside this folder. 

From a terminal prompt, while inside the directory that contains "segmentation_school.py" (i.e., where the HAIL codes are downloaded), call the following command:

    python segmentation_school.py --option predict --project your_project_name --one_network True --encoder_name deeplab --classNum 4 --boxSizeHR 3000 --overlap_percentHR 0.5

Adapt the boxSizeHR parameter to a lower value if your GPU does not have enough memory, and use --gpu 0, --gpu 1, etc to change which device on your computer to use for network prediction (out of range device numbers default to the CPU). When the algorithms have finished processing a slide, a new folder will be created: your_project_name/TRAINING_data/0/Predicted_XMLs. Inside, there will be .xml files which correspond to whole slide predictions for each .svs file inside the TRAINING_data/0/ folder.

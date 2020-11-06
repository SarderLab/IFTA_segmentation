# IFTA_segmentation

This repository contains shared materials for the JASN manuscript entitled Automated Computational Detection of Interstitial Fibrosis, Tubular Atrophy, and Glomerulosclerosis, submitted to JASN in November 2020. Shared materials include example whole slide images and their CNN segmented output, and codes and pretrained CNN  model to perform IFTA and glomerulosclerosis segmentation on new whole slide images


# Whole slide images
To view the CNN segmentationon whole slide images, you must have Aperio ImageScope (https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) installed on your computer. Then, simply place the whole slide image (.svs file) and the annotations (.xml file) in the same directory on your computer. Upon opening the .svs file with ImageScope, you will see the CNN predictions overlaid on the whole slide. 


# Performing segmentation on new data
Before segmenting your own whole slides, you will need to:

1) Configure HAIL (https://github.com/SarderLab/H-AI-L) and its dependencies on your system that will perform the segmentation
2) Download the pre-trained model file available at https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e 

HAIL operates by performing segmentation at the whole slide level, and takes whole slide images as input. If you do not have your own whole slide image data, sample data can be found here https://buffalo.box.com/s/thlo5vry0ii8sutvke9bmva0gm5aos0e, along with pre-generated network segmentation outputs.




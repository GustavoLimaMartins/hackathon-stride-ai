
Hackathon - v5 2026-07-11 7:03pm
==============================

This dataset was exported via roboflow.com on July 12, 2026 at 1:23 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 261 images.
Hackathon are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Grayscale (CRT phosphor)

The following augmentation was applied to create 10 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Randomly crop between 0 and 15 percent of the image
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 1.05 percent of pixels



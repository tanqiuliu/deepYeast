# deepYeast
Fully Automated Tool for Yeast Cell Detection and Segmentation in Microscopic Images
###### Tanqiu Liu

## Introduction
In biological research, even though the experiment is applied to a number of cells equally, the responses of cells varies greatly. Therefore, we usually want to extract single cell data. For fluorescnt microscopy experiments, we may want to record the dynamics a certain kind of protein in a single cell. It is required to do the segmentation before data extraction to detect sub-regions of an image which represent single cells. **deepYeast** is develop to address the problem of detecting the yeast cell contour.

A sample image：

![](./markdown/example1.tif)

## Methods
The detection process consists of 4 steps: 1) Preprocessing of images, 2) locating cell centers with a convolutional neural network(CNN), 3) detecting cell contour.

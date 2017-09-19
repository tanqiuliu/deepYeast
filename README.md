# deepYeast
Fully Automated Tool for Yeast Cell Detection and Segmentation in Microscopic Images
###### Tanqiu Liu

## Introduction
In biological research, even though the experiment is applied to a number of cells equally, the responses of cells varies greatly. Therefore, we usually want to extract single cell data. For fluorescnt microscopy experiments, we may want to record the dynamics a certain kind of protein in a single cell. It is required to do the segmentation before data extraction to detect sub-regions of an image which represent single cells. **deepYeast** is develop to address the problem of detecting the yeast cell contour.

A sample image：

<img src="./markdown/example1.tif" width = "256" height = "256" alt="sample image" align=center />


## Methods
The detection process consists of 2 steps: 1) locating cell centers with a convolutional neural network(CNN), 2) detecting cell contour.

###1. Locating cell centers with a convolutional neural network(CNN)
A convolutional neural network model was developed to automatically identify the location of the cell centers. The pipeline employed a sliding window approach for detection. That is , a small window slides across the entire image and for position output whether the position is the center of a cell or not. 
The training data was extracted and transformed from raw .csg files which are manually labeled by former researchers. Positive examples are obtained by generating a window centered on the center of the cell. Negative examples are obtained by random sampling outside the cells.

<img src="./markdown/WX20170918-205841@2x.png" alt="training examples" align=center>

The convolutional neural network consists of three convolutional layers and two dense layers. The CNN predicts whether the current window is a cell. 

<img src="./markdown/WX20170918-205912@2x.png" alt="CNN structure" align=center>

For each image, the sliding window method predicts a set of points which are candidates of cell centers. Following postprocessing of these candidates includes discard points outside the cells and merge points in the cells. Finally we get precise and robust predictions of cell centers.

<img src="./markdown/Picture3.png" >    <img src="./markdown/Picture4.png" >    <img src="./markdown/Picture5.png" >


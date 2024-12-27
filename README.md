# Image Captioning
Automatically Generating Captions for Images
##  Overview
In this Project we have to combine Deep Convolutional Nets for image classification  with Recurrent Networks for sequence modeling, to create a single network that generates descriptions of image using [Flickr Dataset]

Flickr is a small image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. GPU Accelerated Computing (CUDA) is neccessery for this project.


## Project Structure
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

__Notebook 0__ : Get (Flickr) dataset ready;

__Notebook 1__ : Test the Flickr dataset;

__Notebook 2__ : Training the CNN-RNN Model;

__Notebook 3__ : Load trained model and generate predictions.


## .py files used in the notebooks
__data loader__ : Make a data loader from the data set made in the Notebook0

__model__ : Create the CNN->LSTM model

__vocabulary__ : Create a vocabulary pkl 


## References
[arXiv:1411.4555v2 [cs.CV] 20 Apr 2015](https://arxiv.org/pdf/1411.4555.pdf) and [arXiv:1502.03044v3 [cs.LG] 19 Apr 2016](https://arxiv.org/pdf/1502.03044.pdf)

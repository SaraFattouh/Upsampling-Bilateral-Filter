# Upsampling Bilateral Filter
Apply a bilateral filter to upsample a depth image, guided by an RGB image.

Upsampling an input image (increasing of the spatial resolution) using a bilateral filter which is a non-linear, edge-preserving, and noise-reducing smoothing filter. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. This weight is based on a Gaussian distributio and it depends not only on Euclidean distance of pixels, but also on the radiometric differences (such as color intensity). This preserves sharp edges.

Parts of the program:
- There is a CMakeLists.txt file for compilation.
- The main.cpp file contains:
  -  Implementation of a bilateral filter. 
  -  Some comparison metrics that can be used for comparing images like SSD, MSE, RMSE and PNSR. 
  -  A function that converts a bilateral filter to Guided Joint bilateral filter for guided image upsampling. 
 
 The main function takes as input:

-   aligned RGB and Depth/Disparity image pair 
  - σs: first parameter of the bilateral filter, spatial filter kernel 
  - σr: second  parameter of the bilateral filter, spectral filter kernel AKA range kernel

The result can be found in the build directory 
  - It contains the resulting upsampled disparity maps obtained by running the Bilateral filter on the image with combinations, 16 in total, of four different levels of sigmas for the spatial and four for the spectral filter.

Image pairs input could be found in Middlebury stereo dataset https://vision.middlebury.edu/stereo/data/ , or any of your choice.



 
![plot](https://github.com/SaraFattouh/Upsampling-Bilateral-Filter/blob/main/upsampling.jpg) 

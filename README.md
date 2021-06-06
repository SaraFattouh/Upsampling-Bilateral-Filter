# Upsampling-Bilateral-Filter
Apply the filter to upsample a depth image, guided by an RGB image

Upsampling an input image (increasing of the spatial resolution) using a bilateral filter which is a non-linear, edge-preserving, and noise-reducing smoothing filter. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. This weight is based on a Gaussian distribution. These weights depend not only on Euclidean distance of pixels, but also on the radiometric differences (such as color intensity). This preserves sharp edges.

Parts of the program:
- There is a CMakeLists.txt file for compilation.
- In the main.cpp file, the main program implementes the bilateral filter, some comparison metrics that can be used for comparing images like SSD, mse, rmse, psnr and convert a bilateral filter to Guided Joint bilateral filter for guided image upsampling. It takes, as input:

-   aligned RGB and Depth/Disparity image pair 
  - σs: first parameter of the bilateral filter, spatial filter kernel 
  - σr: second  parameter of the bilateral filter, spectral filter kernel AKA range kernel

- The result can be found in the build directory 
  - It contains the resulting upsampled disparity maps obtained by running the Bilateral filter on the image with combinations, 16 in total, of four different levels of sigmas for the spatial and four for the spectral filter.
 

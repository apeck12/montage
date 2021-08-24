# montage
This package supports montage cryo-electron tomography (cryoET) as a technique to image a large field-of-view at high resolution. 

The code in the `collect’ folder enables scoring the efficiency of different tiling strategies by simulation and writing the desired strategy to an input file for SerialEM. Here we assume that data will be collected using a circular beam equipped with fringe-free illumination.

The `process’ folder contains a series of scripts for assembling the acquired data into a montage tilt-series. Corrections during data pre-processing include Fresnel fringe removal, adjustment for uneven radial illumination, and CTF correction. Non-python dependencies include IMOD and CTFFIND4. 

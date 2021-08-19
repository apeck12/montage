# Script for running ctfphaseflip to correct defocus, which takes as command line input:
# [input_file] [output_file] [tilt_angle_file] [defocus_file]
# Default settings are set up such that each image is corrected based on a single defocus
# value, without strip-based correction.

# common microscope parameters
pixel_size=0.265 # in nm
voltage=300
Cs=2.70
amp_contrast=0.07
def_tolerance=25
interp_width=1
max_width=5760
axis_angle=86.1

ctfphaseflip -input $1 -output $2 -angleFn $3 -defFn $4 -defTol ${def_tolerance} -iWidth ${interp_width} -maxWidth ${max_width} -pixelSize ${pixel_size} -volt ${voltage} -cs ${Cs} -ampContrast ${amp_contrast} -AxisAngle ${axis_angle}

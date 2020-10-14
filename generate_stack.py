from collections import OrderedDict
from natsort import natsorted
import mrcfile, argparse
import glob, time, os, utils
import numpy as np

"""
Compile all stitched tiles together into a tilt-series and save in MRC format.
"""

############################################ 
# Parsing and modifying command line input #
############################################

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Generate a tilt-series from a set of sttiched tiles.')
    parser.add_argument('-i','--stitched_prefix', help='Path to prefix file name of stitched images', 
                        required=True, type=str)
    parser.add_argument('-v','--voxel_path', help='Path to MRC file with voxel size in header',
                        required=True, type=str)
    parser.add_argument('-c','--centers', help='Path to input beam centers file',
                        required=True, type=str)
    parser.add_argument('-o','--output', help='Path to which to save tilt stack',
                        required=True, type=str)
    parser.add_argument('-w','--width', help='Width along x and y of final montage',
                        required=True, type=int)
    parser.add_argument('-r','--rotation', help='Global rotation to apply to all beam centers',
                        required=False, type=float, default=0)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    
    mrc = mrcfile.open(args['voxel_path'])
    args['voxel_size'] = float(mrc.voxel_size.x) # Angstrom / pixel
    mrc.close()

    return args


######################################################
# Set-up: retrieving beam centers, applying rotation #
######################################################

def retrieve_beam_centers(centers_file, voxel_size):
    """
    Retrieve the position of the central tile for each tilt angle and convert from 
    microns to pixels
    
    Inputs:
    -------
    centers_file: filename of predicted beam centers, where x,y coordinates of each 
        tile are listed on separate lines after a particular tilt angle
    voxel_size: in A/pixel to convert coordinates from microns to pixels
    
    Outputs:
    --------
    origin_shifts: 2d array whose nth row gives the (x,y) origin shift of the nth tilt
    tilt_angles: ordered array of tilt angles
    """
    
    origin_shifts, tilt_angles = list(), list()
    f = open(centers_file, 'r') 
    content = f.readlines() 
    
    # extract positoin of tile 0 for each tilt angle
    for line in content:
        as_array = np.array(line.strip().split()).astype(float)
        if (len(as_array) == 1):
            tilt_angles.append(as_array[0])
            counter = 0
        elif (len(as_array) >= 2) and (counter==0):
            origin_shifts.append(as_array * 1e4/voxel_size)
            counter += 1
            
    origin_shifts = np.array(origin_shifts)[:,:2]     
    return origin_shifts, np.array(tilt_angles)


def apply_rotation(beam_centers, rotation_angle):
    """
    Rotate beam centers in plane of detector.
    
    Inputs:
    -------
    beam_centers: 2d array of tiles' center coordinates 
    rotation_angle: rotation angle in degrees
    
    Outputs:
    --------
    r_beam_centers: rotated beam centers
    """
    theta=np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    return beam_centers.dot(R)


#################################################
# Process all montaged images into a tilt-stack #
#################################################

def stack_stitched(args):
    """
    Stack stitched tilt images into a tilt-series.

    Inputs:
    -------
    args: dict of command line arguments

    Outputs:
    --------
    tilt_series: 3d array of tilt-series
    """
    # retrieve origin shifts and tilt angles
    shifts, tilts = retrieve_beam_centers(args['centers'], args['voxel_size'])
    shifts = apply_rotation(shifts, args['rotation'])
    shifts = np.fliplr(shifts)
    t_shifts = OrderedDict((key,val) for key,val in zip(tilts,shifts))
    
    # generate tilt-series
    tilt_series = np.zeros((len(tilts),args['width']*2,args['width']*2))
    for xi,t in enumerate(sorted(tilts)):
        image = mrcfile.open(args['stitched_prefix'] + f"{int(t)}.mrc").data
        xc, yc = (np.array(image.shape)/2).astype(int)
        x_offset, y_offset = np.array(t_shifts[t]).astype(int)
        tilt_series[xi] = image[xc-args['width']-x_offset: xc+args['width']-x_offset,
                                yc-args['width']-y_offset: yc+args['width']-y_offset]

    return tilt_series


#################################################
# Generate tilt stack from stitched tilt images #
#################################################

if __name__ == '__main__':

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    tseries = stack_stitched(args)
    utils.save_mrc(tseries, args['output'], args['voxel_size'])

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

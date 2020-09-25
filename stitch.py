from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import mrcfile, itertools, pickle
import numpy as np
import numpy.ma as ma

"""
Stitch tiles into an MRC file based on input beam centers and Fresnel fringe masks.
"""

############################################ 
# Parsing and modifying command line input #
############################################

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Stitch all tiles for a given tilt angle.')
    parser.add_argument('-i','--image_paths', help='Path to images in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-m','--mask_paths', help='Path to masks in glob-readable format',
                        required=True, type=str)
    parser.add_argument('-c','--centers', help='Path to optimized beam centers in pickle format',
                        required=True, type=str)
    parser.add_argument('-o','--output', help='Savename of output mrc file',
                        required=True, type=str)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    # expand paths keys into ordered lists
    for key in ['image_paths', 'mask_paths']:
        args[key] = natsorted(glob.glob(args[key]))

    mrc = mrcfile.open(args['image_paths'][0])
    args['voxel_size'] = float(mrc.voxel_size.x) # Angstrom / pixel
    mrc.close()
    
    return args


##########################################
# Stitching and conversion to mrc format #
##########################################

def save_mrc(volume, savename):
    """
    Save Nd numpy array, volume, to path savename in mrc format.
    
    Inputs:
    -------
    volume: Nd array to be saved
    savename: path to which to save Nd array in mrc format
    """
    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(volume.astype(np.float32))
    mrc.close()
    return


def stitch(image_paths, mask_paths, centers, savename=None):
    """
    Stitch tiles together into a single array based on input centers, averaging
    the values of pixels that overlap. Optionally save as an mrc file.
    
    Inputs:
    -------
    image_paths: ordered list of tile file names
    mask_paths: ordered list of mask file names
    centers: 2d array of optimized beam coordinates
    savename: if provided, save as mrc file
    
    Outputs:
    --------
    stitched: array of stitched images
    """
    
    # generate a canvas slightly larger than required
    tile_shape = mrcfile.open(image_paths[0]).data.shape
    canvas_size = np.array(centers.max(axis=0)-centers.min(axis=0)+tile_shape, dtype=int)
    canvas_size  = np.array(canvas_size*1.1, dtype=int) 
    
    # set up canvas that images will be added to and center coordinates within canvas frame
    stitched = np.zeros(tuple(canvas_size))
    counts = np.zeros(tuple(canvas_size))
    COM = np.array((canvas_size/2*centers.shape[0] - centers.sum(axis=0))/centers.shape[0], dtype=int)
    centers += COM
    
    # add tiles to canvas and divde by number of unmasked pixels
    Rx,Ry = tile_shape
    upleft = np.array(centers - [Rx/2, Ry/2],int)
    for i in range(centers.shape[0]):
        tile = mrcfile.open(image_paths[i]).data.copy()
        mask = np.load(mask_paths[i])
        tile *= mask
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        stitched[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += tile 
        counts[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask
    stitched[counts!=0] /= counts[counts!=0]
    
    if savename is not None:
        save_mrc(stitched, savename)
        
    return stitched


#################################################
# Stitch all tiles from a particular tilt image #
#################################################

if __name__ == '__main__':

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    opt_centers = pickle.load(open(args['centers'], "rb"))
    centers = np.array(list(opt_centers.values()))
    stitch(args['image_paths'], args['mask_paths'], centers, savename=args['output'])

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

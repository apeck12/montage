from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import mrcfile, itertools, pickle
import numpy as np
import numpy.ma as ma
import utils, scipy.ndimage

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
    parser.add_argument('-s','--strategy', help='Strategy for overlap region: nearest or average',
                        required=True, type=str)
    parser.add_argument('-o','--output', help='Savename of output mrc file',
                        required=True, type=str)
    parser.add_argument('-A','--A', help='Pickle file of 2d affine matrix for each tile (no translations)',
                        required=False, type=str)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    # expand paths keys into ordered lists
    for key in ['image_paths', 'mask_paths']:
        args[key] = natsorted(glob.glob(args[key]))

    # retrieve optimized centers
    opt_centers = pickle.load(open(args['centers'], "rb"))
    args['centers'] = np.array(list(opt_centers.values()))

    # set up affine transformation details if provided
    if args['A'] is not None:
        args['A'] = pickle.load(open(args['A'], "rb"))
        args['offsets'] = OrderedDict((key,np.zeros(2)) for key in args['A'].keys())
    else:
        args['offsets'] = None

    return args


##########################################
# Stitching and conversion to mrc format #
##########################################


def stitch_average(image_paths, mask_paths, centers, matrices=None, offsets=None, savename=None):
    """
    Stitch tiles together into a single array based on input centers, averaging
    the values of pixels that overlap. Optionally save as an mrc file.
    
    Inputs:
    -------
    image_paths: ordered list of tile file names
    mask_paths: ordered list of mask file names
    centers: 2d array of optimized beam coordinates
    matrices: dict of affine transformation matrices for each tile, optional
    offsets: dict of translational offsets for each tile, optional
    savename: if provided, save as mrc file
    
    Outputs:
    --------
    stitched: array of stitched images
    """
    # round centers to start
    centers = np.around(centers).astype(int)
    if offsets is None:
        offsets = OrderedDict((key,np.zeros(2)) for key in range(centers.shape[0]))
    
    # generate a canvas slightly larger than required
    tile_shape = mrcfile.open(image_paths[0]).data.shape
    canvas_size = np.array(centers.max(axis=0)-centers.min(axis=0)+tile_shape, dtype=int)
    canvas_size  = np.array(canvas_size*1.2, dtype=int) 
    
    # set up canvas that images will be added to and center coordinates within canvas frame
    stitched = np.zeros(tuple(canvas_size))
    counts = np.zeros(tuple(canvas_size))
    COM = np.array((canvas_size/2*centers.shape[0] - centers.sum(axis=0))/centers.shape[0], dtype=int)
    centers += COM
    
    # add tiles to canvas and divde by number of unmasked pixels
    Rx,Ry = tile_shape
    upleft = np.around(np.array(centers - [Rx/2, Ry/2])).astype(int)

    for i in range(centers.shape[0]):
        tile, mask = utils.load_mask_tile(image_paths[i], mask_paths[i], as_masked_array=False)
        if matrices is not None:
            tile = scipy.ndimage.interpolation.affine_transform(tile, matrices[i], offsets[i])
            mask = scipy.ndimage.interpolation.affine_transform(mask.astype(int), matrices[i], offsets[i])
        mask = np.around(mask).astype(int)
        tile *= mask
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        stitched[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += tile 
        counts[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask

    stitched[counts!=0] /= counts[counts!=0]  
    
    if savename is not None:
        utils.save_mrc(stitched, savename)
        
    return stitched


def stitch_nearest(image_paths, mask_paths, centers, matrices=None, offsets=None, savename=None):
    """
    Stitch tiles together into a single array based on input centers. For pixels
    in overlap regions, choose the value from the tile that is closest. 
    
    Inputs:
    -------
    image_paths: ordered list of tile file names
    mask_paths: ordered list of mask file names
    centers: 2d array of optimized beam coordinates
    matrices: dict of affine transformation matrices for each tile, optional
    offsets: dict of translational offsets for each tile, optional
    savename: if provided, save as mrc file
    
    Outputs:
    --------
    stitched: array of stitched images
    """
    import scipy.spatial
    
    # round centers to start
    centers = np.around(centers).astype(int)
    if offsets is None:
        offsets = OrderedDict((key,np.zeros(2)) for key in range(centers.shape[0]))
    
    # generate a canvas slightly larger than required
    tile_shape = mrcfile.open(image_paths[0]).data.shape
    canvas_size = np.array(centers.max(axis=0)-centers.min(axis=0)+tile_shape, dtype=int)
    canvas_size  = np.array(canvas_size*1.2, dtype=int) 
    
    # set up canvas that images will be added to and center coordinates within canvas frame
    counts = np.zeros(tuple(canvas_size))
    COM = np.array((canvas_size/2*centers.shape[0] - centers.sum(axis=0))/centers.shape[0], dtype=int)
    centers += COM
    
    # add masks to canvas to track which pixels belong to montage
    Rx,Ry = tile_shape
    upleft = np.around(np.array(centers - [Rx/2, Ry/2])).astype(int)

    for i in range(centers.shape[0]):
        if mask_paths[i][-3:] == 'mrc':
            mask = mrcfile.open(mask_paths[i]).data.astype(bool)
        else:
            mask = np.load(mask_paths[i])
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        counts[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask

    # determine which tile is nearest to all montage pixels using counts array
    overlap_idx = np.array(np.where(counts>0))
    dist = scipy.spatial.distance.cdist(overlap_idx.T, centers)
    min_dist = np.argmin(dist, axis=1) + 1
    
    nearest_tile = np.zeros_like(counts)
    nearest_tile[counts>0] = min_dist
    
    # add images to canvas, optionally applying an affine transformation
    stitched = np.zeros(tuple(canvas_size))
    for i in range(centers.shape[0]):
        prelim = np.zeros_like(counts)
        tile, mask = utils.load_mask_tile(image_paths[i], mask_paths[i], as_masked_array=False)
        if matrices is not None:
            tile = scipy.ndimage.interpolation.affine_transform(tile, matrices[i], offsets[i])
            mask = scipy.ndimage.interpolation.affine_transform(mask.astype(int), matrices[i], offsets[i])
        mask = np.around(mask).astype(int)
        tile *= mask
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        prelim[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += tile 
        prelim[nearest_tile!=(i+1)] = 0
        stitched += prelim
    
    if savename is not None:
        utils.save_mrc(stitched, savename)
        
    return stitched


#################################################
# Stitch all tiles from a particular tilt image #
#################################################

if __name__ == '__main__':

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    if args['strategy'] == 'average':
        stitch_average(args['image_paths'], args['mask_paths'], args['centers'], 
                       matrices=args['A'], offsets=args['offsets'], savename=args['output'])
    elif args['strategy'] == 'nearest':
        stitch_nearest(args['image_paths'], args['mask_paths'], args['centers'],
                       matrices=args['A'], offsets=args['offsets'], savename=args['output'])
    else:
        print("Stitching strategy not recognized. Should be nearest or average")

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

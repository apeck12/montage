from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import mrcfile, itertools, pickle
import numpy as np
import utils

"""
Stitch all tiles for a given tilt angle into a montaged image based on input
beam centers and applying given masks. For pixels in the overlap region, the
intensity value is taken from the tile whose center is nearest. Any gaps are
filled by sampling intensity values from the neighboring region.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Stitch all tiles for a given tilt angle.')
    parser.add_argument('-i','--image_paths', help='Path to images in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-m','--mask_paths', help='Path to masks in glob-readable format',
                        required=True, type=str)
    parser.add_argument('-c','--centers', help='Path to optimized beam centers in pickle or numpy format',
                        required=True, type=str)
    parser.add_argument('-p','--params', help='Path to parameters describing (h,k,r) of each circular tile',
                        required=True, type=str)
    parser.add_argument('-l','--length', help='Box length for filling in missing pixels by local sampling',
                        required=True, type=int)
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

    # retrieve optimized centers
    if args['centers'][-3:] == 'npy':
        args['centers'] = np.load(args['centers'])
    else:
        opt_centers = pickle.load(open(args['centers'], "rb"))
        args['centers'] = np.array(list(opt_centers.values()))
        
    # retrieve parameters
    args['params'] = np.load(args['params'])[:,:2]
    assert args['centers'].shape == args['params'].shape

    # retrieve voxel size from MRC header in Angstrom per pixel
    mrc = mrcfile.open(args['image_paths'][0])
    args['voxel_size'] = float(mrc.voxel_size.x)
    mrc.close()

    return args


def preprocess_tile(image_path, mask_path):
    """
    Apply gain correction and mask to tile.
    
    Parameters
    ----------
    image_path : string 
        path to tile
    mask_path : string 
        path to mask containing gain correction and fringe mask
    
    Returns
    -------
    tile : numpy.ndarray, shape (N,M)
        gain-corrected, masked tile with same shape as input image
    """
    # load tile and convert from int to float
    mrc_tile = mrcfile.open(image_path)
    tile = mrc_tile.data.copy().astype(float)
    mrc_tile.close()

    # retrieve gain correction and fresnel mask
    mrc_mask = mrcfile.open(mask_path)
    gain_corr = mrc_mask.data.copy()
    mrc_mask.close()
    
    # apply gain correction, optionally bandpass filter, and generate masked array
    tile *= gain_corr
    return tile


def fill_gaps_by_sampling(image, missing_idx, length):
    """
    Fill missing pixels by randomly sampling from neighboring intensity values.
    
    Parameters:
    -------
    image : numpy.ndarray, shape (N,M)
        stitched image with gaps to be filled
    missing_idx : numpy.ndarray, shape (N,2)
        coordinates of pixels that need to be filled by sampling
    length : int 
        length of square region around missing pixels to sample from
    
    Returns:
    --------
    image : numpy.ndarray, shape (N,M)
        stitched image with gaps filled
    """

    hlength = int(length/2)
    for idx in missing_idx:
        inset = image[idx[0]-hlength:idx[0]+hlength, idx[1]-hlength:idx[1]+hlength].copy()
        mask = np.ones_like(inset)
        adjusted_idx = missing_idx - idx + hlength
        adjusted_idx = adjusted_idx[(adjusted_idx[:,0]>=0) & (adjusted_idx[:,0]<length)]
        adjusted_idx = adjusted_idx[(adjusted_idx[:,1]>=0) & (adjusted_idx[:,1]<length)]   
        mask[tuple(adjusted_idx.T)] = 0 
        image[tuple(idx.T)] = np.random.choice(inset[mask!=0])
    
    return image


def stitch(image_paths, mask_paths, centers, params, length, voxel_size=None, savename=None):
    """
    Stitch tiles together into a single array based on input centers. For pixels
    in overlap regions, choose the value from the tile that is closest. Any gaps
    are filled by choosing the value from the next closest tile; if gaps remain,
    then the values of missing pixels are filled by sampling from nearby pixels.
    
    Parameters
    ----------
    image_paths : list of strings
        filenames of tiles to stitch
    mask_paths : list of strings
        filenames of masks, ordered as image_paths
    centers : numpy.ndarray, shape (N, 2)
        optimized beam coordinates relative to the central tile, ordered as image_paths
    params : numpy.ndarray, shape (N, 2)
        optimized beam centers in the frame of the detector, ordered as image_paths 
    length : int 
        box length for filling in missing pixels by sampling
    voxel_size : string, optional
        voxel size for MRC header; if None, do not amend header
    savename : string, optional
        path to MRC file to save; if None, stitch is not saved
    
    Returns
    -------
    stitched : numpy.ndarray, shape (N, N)
        stitched projection image
    """
    import scipy.spatial
    
    # round centers to start
    centers = np.around(centers).astype(int)
    
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
        mask = mrcfile.open(mask_paths[i]).data.copy()
        mask[mask!=0] = 1
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        counts[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask

    # actual centers of illuminated region of each tile
    offsets = params - [Rx/2, Ry/2]
    true_centers = centers.copy() + offsets 

    # determine which tile is nearest to all montage pixels using counts array
    overlap_idx = np.array(np.where(counts>0))
    dist = scipy.spatial.distance.cdist(overlap_idx.T, true_centers)
    min_dist = np.argmin(dist, axis=1) + 1
    
    nearest_tile = np.zeros_like(counts)
    nearest_tile[counts>0] = min_dist
    
    # add images to canvas; overlap pixels are filled from nearest tile
    stitched = np.zeros(tuple(canvas_size))
    for i in range(centers.shape[0]):
        prelim = np.zeros_like(counts)
        tile = preprocess_tile(image_paths[i], mask_paths[i])
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        prelim[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += tile 
        prelim[nearest_tile!=(i+1)] = 0
        stitched += prelim

    d_max = dist.max()
    dist[np.arange(len(dist)), dist.argmax(1)] = d_max

    # generate stitch of where counts were actually found
    actual_counts = np.zeros(tuple(canvas_size))
    for i in range(centers.shape[0]):
        prelim = np.zeros_like(counts)
        mask = mrcfile.open(mask_paths[i]).data.copy()
        mask[mask!=0] = 1
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        assert(upleft_x >= 0 and upleft_y >= 0)
        prelim[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask
        prelim[nearest_tile!=(i+1)] = 0
        actual_counts += prelim

    # generate rough stitch of where counts were expected
    centers_dist = scipy.spatial.distance.cdist(np.array([centers[0]]), centers)[0]
    centers_dist -= centers_dist.max()
    inner_tiles = np.where(np.abs(centers_dist)>0.5*Rx)[0]
    
    expected_counts = np.zeros(tuple(canvas_size))
    hRx = int(Rx/2)
    for i in inner_tiles:
        xc,yc = centers[i]
        expected_counts[xc-hRx:xc+hRx,yc-hRx:yc+hRx] = 1
    
    # find and fill in gaps
    gap_idx = np.array(np.where((actual_counts==0) & (expected_counts>0)))
    if len(gap_idx[0]) > 0:
        print(f"Warning: {len(gap_idx[0])} missing pixels, attempting to patch")
        stitched = fill_gaps_by_sampling(stitched, gap_idx.T, length)

    # optionally save as an MRC file
    if savename is not None:
        utils.save_mrc(stitched, savename, voxel_size=voxel_size)
        
    return stitched


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)
    stitched = stitch(args['image_paths'], 
                      args['mask_paths'], 
                      args['centers'], 
                      args['params'],
                      args['length'],
                      voxel_size=args['voxel_size'], 
                      savename=args['output'])

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f} minutes")


if __name__ == '__main__':
    main()

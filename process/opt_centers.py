from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import itertools, pickle, sys
import numpy as np
import numpy.ma as ma
import scipy.optimize, skimage.filters
import mrcfile, utils

"""
Determine optimal positions of tile centers for a tilt angle of interest. Optimized
positions are relative to the central tile, which is fixed at the origin.
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
    parser.add_argument('-c','--centers', help='Path to input or optimized beam centers file',
                        required=True, type=str)
    parser.add_argument('-t','--tilt_angle', help='Tilt angle to process',
                        required=True, type=int)
    parser.add_argument('-b','--beam_diameter', help='Beam diameter in microns',
                        required=True, type=float)
    parser.add_argument('-ms','--max_shift', help='Maximum shift per grid search in pixels',
                        required=False, type=int, default=25)
    parser.add_argument('-o','--save_dir', help='Directory to which to save output files',
                        required=True, type=str)
    parser.add_argument('-r','--rotation', help='Global rotation to apply to all beam centers',
                        required=False, type=float, default=0)
    parser.add_argument('-bf','--bin_factor', help='Factor by which optimized centers should be unbinned',
                        required=False, type=float)
    parser.add_argument('-n','--max_iter', help='Number of iterations for each CC grid search',
                        required=False, type=int, default=5)
    parser.add_argument('-nl','--n_layers', help='Number of hexagonal layers, excluding central tile',
                        required=False, type=int, default=3)
    parser.add_argument('-bp','--bandpass', help="Low, high kernel sizes, percentile threshold for bandpass filter",
                        required=False, type=float, nargs=3)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    # expand paths keys into ordered lists
    for key in ['image_paths', 'mask_paths']:
        args[key] = natsorted(glob.glob(args[key]))

    # check that the numbers of tiles and masks match
    if len(args['image_paths']) != len(args['mask_paths']):
        print("Warning! Numbers of tiles and masks do not match; cannot proceed with optimizing centers.")
        sys.exit()

    # retrieve voxel size from MRC header in Angstrom per pixel
    mrc = mrcfile.open(args['image_paths'][0])
    args['voxel_size'] = float(mrc.voxel_size.x)
    mrc.close()

    # create output directory as needed
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    if args['bandpass'] is not None:
        args['low_sigma'], args['high_sigma'], args['p_threshold'] = args['bandpass']
    else:
        args['low_sigma'], args['high_sigma'], args['p_threshold'] = None, None, None

    return args


def retrieve_beam_centers(centers_file, tilt_angle, voxel_size=None):
    """
    Retrieve predicted beam centers for given tilt_angle from the file used for data 
    collection. If an mrc file is provided, convert the positions from um to pixels.
    
    Parameters
    ----------
    centers_file : string 
        filename of predicted beam centers, where x,y coordinates of each 
        tile are listed in um on separate lines after the relevant tilt angle
    tilt_angle : float
        tilt angle of interest
    voxel_size : float 
        voxel dimensions in A/pixel; if supplied, convert coordinates from microns to pixels 
    
    Returns
    -------
    beam_centers : numpy.ndarray, shape (N, 2)
        unoptimized, relative beam centers of each tile collected at the tilt angle of interest
    """
    
    f = open(centers_file, 'r') 
    content = f.readlines() 
    beam_centers = list()

    # extract beam coordinates for tilt angle of interest
    records = False
    for line in content:
        as_array = np.array(line.strip().split()).astype(float)
        if (len(as_array) == 1) and (as_array[0] == tilt_angle):
            records = True
        elif (len(as_array) >= 2) and (records is True):
            beam_centers.append(as_array)
        elif (len(as_array) == 1) and (as_array[0] != tilt_angle):
            records = False
    beam_centers = np.array(beam_centers)[:,:2]
    
    # convert from microns to pixels if mrcfile is provided
    if voxel_size is not None:
        beam_centers *= 1e4/voxel_size
        
    return beam_centers


def predict_overlaps(beam_centers, beam_diameter, voxel_size, n_layers):
    """
    Predict the indices of overlapping tiles.
    
    Parameters
    ----------
    beam_centers : numpy.ndarray, shape (N, 2)
        unoptimized, relative beam centers of each tile in pixels
    beam_diamter : float
        diamter of beam in microns
    voxel_size : float 
        voxel dimension in Angstrom per pixel
    n_layers : int 
        number of hexagonal layers in montage, not including central tile
    
    Returns
    -------
    d_overlap : dictionary 
        tile index: array of indices of overlapping tiles
    layer_range : dictionary 
        layer number: array of tile indices belonging to layer
    """
    import scipy.spatial
    
    max_dist = beam_diameter * 1e4 / voxel_size # convert from um to pixels
    dists = scipy.spatial.distance.cdist(beam_centers, beam_centers)
    
    overlap = OrderedDict()
    for i in range(dists.shape[0]):
        preliminary = np.where(dists[i]<max_dist)[0]
        overlap[i] = np.array([xi for xi in preliminary if xi!=i])
        
    layer_range = dict()
    for i in range(n_layers):
        layer_range[i+1] = np.where((dists[0]>i*max_dist) & (dists[0]<(i+1)*max_dist))[0]
        
    return overlap, layer_range


def retrieve_pairs(overlaps, l_range):
    """
    Retrieve relevant pairs of tiles for stitching a particular layer, with each pair 
    formatted as (idx1,idx2), where idx1 corresponds to tiles in a preceding (inner) 
    layer and is fixed. Only pairs of tiles within the layer of interest or a previously
    considered layer are included.
    
    Parameters
    ----------
    d_overlap : dictionary 
        tile index: array of indices of overlapping tiles
    layer_range : dictionary 
        layer number: array of tile indices belonging to layer
    
    Returns
    -------
    layer_pairs : list of tuples
        indices of pairs of overlapping tiles for layer of interest
    """
    
    layer_pairs = list()
    for key1 in overlaps.keys():
        for key2 in overlaps[key1]:
            if key1<=l_range.max() and key2>=l_range.min() and key2<=l_range.max():
                layer_pairs.append((key1,key2))
    
    return layer_pairs


def set_up(args):
    """
    Set up for optimization: retrieve input beam centers, center by positioning tile 0
    on the origin, and rotate as needed. Also determine indices of tiles that overlap.

    Parameters
    ----------
    args : dictionary 
        command line arguments

    Returns
    --------
    tile_centers : numpy.ndarray, shape (N, 2)
        unoptimized, relative beam centers of each tile collected at the tilt angle of interest
        modified for use as initial guesses -- i.e., flipped and potentially rotated
    overlaps : dictionary 
        tile index: array of indices of overlapping tiles
    layer_range : dictionary 
        layer number: array of tile indices belonging to layer
    """
    # tile positions are estimated from inputs to SerialEM
    if args['centers'][-3:] == 'txt':
        tile_centers = retrieve_beam_centers(args['centers'], args['tilt_angle'], voxel_size=args['voxel_size'])
        tile_centers -= tile_centers[0]
        tile_centers = utils.apply_rotation(tile_centers, args['rotation'])
        tile_centers = np.fliplr(tile_centers)
    
    # tile positions have been pre-optimized from binned data
    else:
        tile_centers = np.load(args['centers'])
        tile_centers -= tile_centers[0]
        if args['bin_factor'] is not None:
            tile_centers *= args['bin_factor']
    
    overlaps, layer_range = predict_overlaps(tile_centers, args['beam_diameter'], 
                                             args['voxel_size'], args['n_layers'])
    
    return tile_centers, overlaps, layer_range


def preprocess_tile(image_path, mask_path, low_sigma=None, high_sigma=None, p_threshold=None):
    """
    Load tile and mask, applying the gain correction and mask to the tile. If low_sigma,
    high_sigma, and p_threshold are supplied, then bandpass filter tile and discard any
    pixels below a certain intensity magnitude threshold.
    
    Parameters
    ----------
    image_path : string
        path to tile in .mrc format
    mask_path : string
        path to mask containing gain correction and fringe mask in .mrc format
    low_sigma : float, optional
        size of smaller kernel for bandpass-filtering; if not provided, do not filter
    high_sigma : float, optional
        size of larger kernel for bandpass-filtering; if not provided, do not filter
    p_threshold: float, optional
        what percentile of high-magnitude features to keep; if not provided, keep all
    
    Returns
    -------
    tile : numpy.masked_array, shape (M,N)
        gain-corrected, optionally bandpass-filtered, masked tile
    """
    # load tile and convert from int to float
    mrc_tile = mrcfile.open(image_path)
    tile = mrc_tile.data.copy().astype(float)
    mrc_tile.close()

    # retrieve gain correction and fresnel mask
    mrc_mask = mrcfile.open(mask_path)
    gain_corr = mrc_mask.data.copy()
    mrc_mask.close()
    mask = gain_corr.copy()
    mask[mask!=0] = 1
    mask = np.invert(mask.astype(bool))
    gain_corr[gain_corr==0] = 1
    
    # apply gain correction and optionally bandpass filter/threshold
    tile *= gain_corr
    if low_sigma and high_sigma:
        tile = skimage.filters.difference_of_gaussians(tile, low_sigma, high_sigma)
        if p_threshold:
            threshold = np.percentile(tile[gain_corr!=0], p_threshold)
            tile[tile>threshold] = 0 # strong features are very negative

    # generate and return masked array
    return ma.masked_array(tile, mask=mask)


def compute_normalized_cc(tile1, tile2, center1, center2):
    """
    Compute the normalized cross-correlation score between two tiles positioned at
    given centers: CC = Sum_i,j ( tile1[i,j] * tile2[i,j] ) / Sum_i,j (1)
    
    Parameters
    ----------
    tile1 : numpy.masked_array, shape (M,N)
        pre-processed tile
    tile2 : numpy.masked_array, shape (M,N)
        pre-processed tile that overlaps with tile1
    center1 : numpy.ndarray of shape (2,)
        (row, col) coordinates of center of tile 1
    center2 : numpy.ndarray of shape (2,)
        (row, col) coordinates of center of tile 2
    
    Returns
    -------
    cc_norm : float
        normalized cross-correlation between overlap of tiles 1 and 2, 0 if no overlap
    """
    # get shape and tile centers information
    m, n = tile1.shape 
    center1_r, center1_c = center1
    center2_r, center2_c = center2

    #number of overlapping rows and columns between the tiles
    nrows = int(m - np.abs(center2_r - center1_r))
    ncols = int(n - np.abs(center2_c - center1_c))

    # otherwise compute normalized CC
    if center2_r < center1_r:
        if center2_c < center1_c: 
            # tile 1 at upper right of tile 2
            cc_matrix = tile2[-nrows:, -ncols:] * tile1[:nrows, :ncols]
            norm = np.sum(np.square(tile2[-nrows:, -ncols:])) * np.sum(np.square(tile1[:nrows, :ncols]))
        else: 
            # tile 1 at upper left of tile 2
            cc_matrix = tile2[-nrows:, :ncols] * tile1[:nrows, -ncols:] 
            norm = np.sum(np.square(tile2[-nrows:, :ncols])) * np.sum(np.square(tile1[:nrows, -ncols:]))
    else:
        if center2_c < center1_c: 
            # tile 1 at lower right of tile 2
            cc_matrix = tile2[:nrows, -ncols:] * tile1[-nrows:, :ncols]
            norm = np.sum(np.square(tile2[:nrows, -ncols:])) * np.sum(np.square(tile1[-nrows:, :ncols]))
        else:
            # tile 1 at lower left of tile 2
            cc_matrix = tile2[:nrows, :ncols] * tile1[-nrows:, -ncols:]
            norm = np.sum(np.square(tile2[:nrows, :ncols])) * np.sum(np.square(tile1[-nrows:, -ncols:]))

    if norm == 0:
        return 0
    else:
        return np.sum(cc_matrix) / np.sqrt(norm)


def grid_search(tile1, tile2, center1, center2, max_shift):
    """
    Optimize center of tile2, keeping center of tile1 fixed, by performing a grid
    search over all integer positions within max_shift pixels along x and y.

    Parameters
    ----------
    tile1 : numpy.masked_array, shape (M,N)
        pre-processed tile
    tile2 : numpy.masked_array, shape (M,N)
        pre-processed tile that overlaps with tile1
    center1 : numpy.ndarray of shape (2,)
        (row, col) coordinates of center of tile 1
    center2 : numpy.ndarray of shape (2,)
        (row, col) coordinates of center of tile 2
    max_shift : int
        maximum possible translation along x or y from center2 in pixels
    
    Returns
    -------
    center2_opt : numpy.ndarray of shape (2,)
        optimized position of center2 relative to fixed center1
    max_score : float
        normalized cross-correlation score associated with center2_opt
    cc_matrix : numpy.ndarray of shape (2*max_shift+1, 2*max_shift+1)
        matrix of normalized cross-correlation values spanning grid search range
    """
    shifts_1d = list(range(-max_shift, max_shift+1))
    all_shifts = list(itertools.product(shifts_1d, shifts_1d))

    cc_scores = np.zeros(len(all_shifts))
    for i,shift in enumerate(all_shifts):
        cc_scores[i] = compute_normalized_cc(tile1, tile2, center1, center2 + shift)
        
    shift, max_score = all_shifts[np.argmax(cc_scores)], cc_scores[np.argmax(cc_scores)]
    center2_opt = center2 + shift
    
    cc_matrix = cc_scores.reshape(len(shifts_1d), len(shifts_1d))
        
    return center2_opt, max_score, cc_matrix


def optimize_pair(image_paths, mask_paths, beam_centers, idx1, idx2, max_shift, max_iter=5, 
                  low_sigma=None, high_sigma=None, p_threshold=None):
    """
    Optimize the coordinates of the second tile (index idx2) for a pair of tiles. 
    If the optimized coordinates are at the edge of the maximal translation region,
    recenter the search box and re-optimize up to five times.
    
    Parameters
    ----------
    image_paths : list of strings
        paths to tiles in .mrc format
    mask_paths : list of strings
        paths to masks containing gain correction and fringe mask, ordered as image_paths
    beam_centers : numpy.ndarray, shape (N, 2)
        unoptimized, relative beam centers of each tile in pixels
    idx1 : int
        index of tile 1, whose position will be kept fixed
    idx2 : int
        index of tile 2, whose position will be optimized
    max_shift: int
        maximum possible translation in pixels permitted during coordinates optimization
    max_iter : int
        maximum number of iterations for recentering search box, default=5
    low_sigma : float, optional
        size of smaller kernel for bandpass-filtering; if not provided, do not filter
    high_sigma : float, optional
        size of larger kernel for bandpass-filtering; if not provided, do not filter
    p_threshold: float, optional
        what percentile of high-magnitude features to keep; if not provided, keep all
    
    Returns
    -------
    center2_opt : numpy.ndarray of shape (2,)
        optimized position of center2 relative to fixed center1
    cc_matrix : numpy.ndarray of shape (2*max_shift+1, 2*max_shift+1)
        matrix of normalized cross-correlation values spanning grid search range
    """
    
    # extract centers and fractional shifts; center1 will be held fixed
    center1, center2 = beam_centers[idx1], beam_centers[idx2]
    offset1, offset2 = np.around(center1) - center1, np.around(center2) - center2

    # filter, mask, and normalize tiles of interest
    tile1 = preprocess_tile(image_paths[idx1], 
                            mask_paths[idx1],
                            low_sigma=low_sigma, 
                            high_sigma=high_sigma, 
                            p_threshold=p_threshold)

    tile2 = preprocess_tile(image_paths[idx2], 
                            mask_paths[idx2],
                            low_sigma=low_sigma, 
                            high_sigma=high_sigma, 
                            p_threshold=p_threshold)
    
    # first iteration of optimizing position of tile2
    center2_opt, max_score, cc_matrix = grid_search(tile1, tile2, center1, center2, max_shift)
    
    # search up to a maximum of five iterations if best position is on border
    n_iter, edge = 1, 3
    while n_iter < max_iter:
        max_idx = np.array(np.where(cc_matrix==cc_matrix.max())).flatten()
        if any(max_idx < edge) or any(max_idx >= max_shift*2 - edge):
            center2_opt, max_score, cc_matrix = grid_search(tile1, 
                                                            tile2, 
                                                            center1, 
                                                            center2_opt,
                                                            max_shift)
            n_iter += 1
        else:
            break
    print(f"Optimization required {n_iter} iterations; maximum score is {max_score:.2f}")

    return center2_opt, cc_matrix 


def optimize_tilt(args, tile_centers, overlaps, layer_range):
    """
    Optimize tile positions for a single tilt image. The position of the central tile 
    (tile 0) is fixed at the origin, and the positions of tiles in each adjacent layer
    is considered in turn until the positions of the outermost layer tiles are computed.
    Positions are estimated as the mean of all estimated pairwise positions, weighted by
    the normalized cross-correlation for the relevant pair.
    
    Parameters
    ----------
    args : dictionary 
        command line arguments
    tile_centers : numpy.ndarray, shape (N, 2)
        unoptimized, relative beam centers of each tile collected at the tilt angle of interest
        modified for use as initial guesses -- i.e., flipped and potentially rotated
    overlaps : dictionary 
        tile index: array of indices of overlapping tiles
    layer_range : dictionary 
        layer number: array of tile indices belonging to layer
    
    Returns
    -------
    updated_centers : numpy.ndarray, shape (N, 2) 
        optimized tile positions
    """
    
    opt_centers, scores = OrderedDict(), OrderedDict()
    updated_centers = tile_centers.copy()
    
    for layer in layer_range.keys():
        # determine centers for all pairwise combinations in layer
        layer_pairs = retrieve_pairs(overlaps, layer_range[layer])
        for pair in layer_pairs:
            idx1, idx2 = pair
            c2_opt, ccmat = optimize_pair(args['image_paths'], 
                                          args['mask_paths'],
                                          updated_centers, idx1, idx2, 
                                          args['max_shift'],
                                          max_iter=args['max_iter'],
                                          low_sigma=args['low_sigma'],
                                          high_sigma=args['high_sigma'],
                                          p_threshold=args['p_threshold'])
            opt_centers[(idx1,idx2)], scores[(idx1,idx2)] = c2_opt, ccmat.max()
    
        # for each tile in layer compute center as CC-weighted average
        for xi in layer_range[layer]:
            rel_keys = [key for key in opt_centers.keys() if key[1]==xi]
            optc_sel = np.array([opt_centers[rk] for rk in rel_keys if scores[rk]>0])
            if len(optc_sel): # make sure have valid centers
                w_sel = np.array([scores[rk] for rk in rel_keys if scores[rk]>0])
                w_sel /= np.sum(w_sel)
                updated_centers[xi] = np.sum(optc_sel.T * w_sel, axis=1)
            else:
                print(f"Warning: no valid positions for tile {xi}")
                        
    avg_disp = np.mean(np.linalg.norm(updated_centers[1:] - tile_centers[1:], axis=1))
    print(f"Mean displacement between estimated and input tile centers is {avg_disp} pixels")

    for d,tag in zip([opt_centers, scores],['centers_pairs','scores']):
        with open(os.path.join(args['save_dir'], f"{tag}_{args['tilt_angle']}.pickle"), 'wb') as handle:
            pickle.dump(d, handle)
    
    return updated_centers


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)
    tile_centers, overlaps, layers = set_up(args)
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    opt_centers = optimize_tilt(args, tile_centers, overlaps, layers)
    np.save(os.path.join(args['save_dir'], f"opt_centers_{args['tilt_angle']}.npy"), opt_centers)

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")


if __name__ == '__main__':
    main()

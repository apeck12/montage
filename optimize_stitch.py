from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import mrcfile, itertools, pickle
import numpy as np
import numpy.ma as ma
import utils

"""
Determine the optimal tile centers for stitching all tiles collected for a single tilt angle
into a composite image. The optimized centers, weights, and CC matrices are saved as pickle
files. This assumes that masks have already been generated to remove the Frensel fringes.
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
    parser.add_argument('-c','--centers', help='Path to input beam centers file',
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
    parser.add_argument('-bf','--bin_factor', help='Bin factor for centers file',
                        required=False, type=float)
    parser.add_argument('-A','--A', help='Pickle file of affine matrix for each tile',
                        required=False, type=str)

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

    if args['A'] is not None:
        args['A'] = pickle.load(open(args['A'], "rb"))

    return args


#############################################################
# Set-up: retrieving beam centers, predicting tile overlaps #
#############################################################

def retrieve_beam_centers(centers_file, tilt_angle, voxel_size=None):
    """
    Retrieve predicted beam centers for given tilt_angle from the file used for data 
    collection. If an mrc file is provided, convert the positions from um to pixels.
    
    Inputs:
    -------
    centers_file: filename of predicted beam centers, where x,y coordinates of each 
        tile are listed on separate lines after a particular tilt angle
    tilt_angle: tilt angle of interest
    voxel_size: in A/pixel; if supplied, convert coordinates from microns to pixels
    
    Outputs:
    --------
    beam_centers: 2d array whose nth row gives the (x,y) coordinates of the nth tile
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


def predict_overlaps(beam_centers, beam_diameter, voxel_size):
    """
    Predict which tiles overlap.
    
    Inputs:
    -------
    beam_centers: np.array of beam center coordinates in pixels
    beam_diamter: diamter of beam in microns
    voxel_size: in Angstrom per pixel
    
    Outputs:
    --------
    d_overlap: dict of tile index: array of indices of overlapping tiles
    """
    import scipy.spatial
    
    max_dist = beam_diameter * 1e4 / voxel_size # convert from um to pixels
    dists = scipy.spatial.distance.cdist(beam_centers, beam_centers)
    
    overlap = OrderedDict()
    for i in range(dists.shape[0]):
        preliminary = np.where(dists[i]<max_dist)[0]
        overlap[i] = np.array([xi for xi in preliminary if xi!=i])
        
    return overlap


def set_up(args):
    """
    Set up for optimization: retrieve input beam centers, center by positioning tile 0
    on the origin, and rotate as needed. Also determine indices of tiles that overlap.

    Inputs:
    -------
    args: dict of command line arguments

    Outputs:
    --------
    tile_centers: array of input beam centers, modified for use as initial guesses
    overlaps: dict of tile index: array of indices of overlapping tiles
    """
    # tile positions are estimated from inputs to SerialEM
    if args['bin_factor'] is None:
        tile_centers = retrieve_beam_centers(args['centers'], args['tilt_angle'], voxel_size=args['voxel_size'])
        tile_centers -= tile_centers[0]
        tile_centers = apply_rotation(tile_centers, args['rotation'])
        tile_centers = np.fliplr(tile_centers)
    
    # tile positions have been pre-optimized from binned data
    else:
        tile_centers = np.array(list(pickle.load(open(args['centers'], "rb")).values()))
        tile_centers -= tile_centers[0]
        tile_centers *= args['bin_factor']
    
    overlaps = predict_overlaps(tile_centers, args['beam_diameter'], args['voxel_size'])
    
    return tile_centers, overlaps

    
################################################
# Optimization of centers by cross-correlation #
################################################
    
def compute_normalized_cc(tile1, tile2, center1, center2):
    """
    Compute the normalized cross-correlation score between two tiles positioned at
    given centers: CC = Sum_i,j ( tile1[i,j] * tile2[i,j] ) / Sum_i,j (1)
    
    Inputs:
    -------
    tile1: first tile in masked array format
    tile2: second tile in masked array format
    center1: (row, col) coordinates of center of tile 1
    center2: (row, col) coordinates of center of tile 2
    
    Outputs:
    --------
    cc_norm: normalized cross-correlation score between overlap of tiles 1 and 2
    npix_overlap: number of overlapping pixels between tiles
    """
    # get shape and tile centers information
    m, n = tile1.shape 
    center1_r, center1_c = center1
    center2_r, center2_c = center2

    #number of overlapping rows and columns between the tiles
    nrows = int(m - np.abs(center2_r - center1_r))
    ncols = int(n - np.abs(center2_c - center1_c))

    # if no overlapping rows and columns, return 0
    if nrows <= 0 or ncols <= 0: 
        return 0

    # otherwise compute normalized CC
    if center2_r < center1_r:
        if center2_c < center1_c: 
            # tile 1 at upper right of tile 2
            cc_matrix = tile2[-nrows:, -ncols:] * tile1[:nrows, :ncols]
        else: 
            # tile 1 at upper left of tile 2
            cc_matrix = tile2[-nrows:, :ncols] * tile1[:nrows, -ncols:]  
    else:
        if center2_c < center1_c: 
            # tile 1 at lower right of tile 2
            cc_matrix = tile2[:nrows, -ncols:] * tile1[-nrows:, :ncols]
        else:
            # tile 1 at lower left of tile 2
            cc_matrix = tile2[:nrows, :ncols] * tile1[-nrows:, -ncols:]

    return np.sum(cc_matrix) / cc_matrix.count(), cc_matrix.count()


def optimize_centers(tile1, tile2, center1, center2, max_shift):
    """
    Optimize center of tile2, keeping center of tile1 fixed, by performing a grid
    search over all integer positions within max_shift pixels along x and y.

    Inputs:
    -------
    tile1: first tile in masked array format
    tile2: second tile in masked array format
    center1: (row, col) coordinates of center of tile 1
    center2: (row, col) coordinates of center of tile 2
    max_shift: maximum possible translation along x or y from center2 in pixels
    
    Outputs:
    --------
    center2_opt: optimized position of center2
    max_score: normalized CC score associated with center2_opt
    cc_matrix: matrix of normalized CC values spanning grid search range
    npix_overlap: number of overlapping pixels for max score
    """
    shifts_1d = list(range(-max_shift, max_shift+1))
    all_shifts = list(itertools.product(shifts_1d, shifts_1d))

    cc_scores, npix_matrix = np.zeros(len(all_shifts)), np.zeros(len(all_shifts))
    for i,shift in enumerate(all_shifts):
        cc_scores[i], npix_matrix[i] = compute_normalized_cc(tile1, tile2, center1, center2 + shift)
        
    shift, max_score = all_shifts[np.argmax(cc_scores)], cc_scores[np.argmax(cc_scores)]
    center2_opt = center2 + shift
    
    npix_overlap = npix_matrix[np.argmax(cc_scores)]
    cc_matrix = cc_scores.reshape(len(shifts_1d), len(shifts_1d))
        
    return center2_opt, max_score, cc_matrix, npix_overlap


def optimize_pair(image_paths, mask_paths, beam_centers, idx1, idx2, max_shift, A=None):
    """
    Optimize the coordinates of the second tile (index idx2) for a pair of tiles. 
    If the optimized coordinates are at the edge of the maximal translation region,
    recenter the search box and re-optimize up to five times.
    
    Inputs:
    -------
    image_paths: ordered list of tile file names
    mask_paths: ordered list of mask file names
    beam_centers: 2d array of predicted beam coordinates
    idx1: index of tile1
    idx2: index of tile2
    max_shift: maximum translation allowed during coordinates optimziation
    A: dict of affine matrix for each tile
    
    Outputs:
    --------
    centers2_opt: optimized coordinates for tile2
    cc_matrix: matrix of normalized cross correlation scores for grid search
    npix_overlap: number of overlapping pixels for centers2_opt
    """
    
    # mask and normalize tiles of interest
    if A is None:
        tile1 = utils.normalize(utils.load_mask_tile(image_paths[idx1], mask_paths[idx1]))
        tile2 = utils.normalize(utils.load_mask_tile(image_paths[idx2], mask_paths[idx2]))
    else:
        tile1, mask = utils.load_mask_tile(image_paths[idx1], mask_paths[idx1], as_masked_array=False)
        tile1 = utils.apply_affine(tile1, mask, A[idx1])
        tile2, mask = utils.load_mask_tile(image_paths[idx2], mask_paths[idx2], as_masked_array=False)
        tile2 = utils.apply_affine(tile2, mask, A[idx2])
    
    # extract centers; center1 will be held fixed
    center1, center2 = beam_centers[idx1], beam_centers[idx2]
    center2_opt, max_score, cc_matrix, npix_overlap = optimize_centers(tile1, tile2, center1, center2, max_shift)
    
    # search up to a maximum of five iterations if best position is on border
    n_iter, max_iter, edge = 1, 5, 3
    while n_iter <= max_iter:
        max_idx = np.array(np.where(cc_matrix==cc_matrix.max())).flatten()
        if any(max_idx < edge) or any(max_idx >= max_shift*2 - edge):
            center2_opt, max_score, cc_matrix, npix_overlap = optimize_centers(tile1,
                                                                               tile2, 
                                                                               center1, 
                                                                               center2_opt, 
                                                                               max_shift)
            n_iter += 1
        else:
            break
    print(f"Optimization required {n_iter} iterations; maximum score is {max_score:.2f}")

    return center2_opt, cc_matrix, npix_overlap


def optimize_all(args, tile_centers, overlaps):
    """
    Optimize all tile centers, spiraling outwards from tile 0. Save results to 
    directory and return information needed specifically for stitching.
    """
    # set up storage dictionaries
    opt_centers, opt_centers_all = OrderedDict(), OrderedDict()
    cc_matrices, weights = OrderedDict(), OrderedDict()

    # loop through all tiles
    for idx2 in overlaps.keys():
        # fix central tile position at origin
        if idx2 == 0:
            opt_centers[idx2] = np.zeros(2)
        
        # center each subsequent tile based on already processed tiles
        else:
            idx_overlap = overlaps[idx2]
            for idx1 in idx_overlap:
                if idx1 in opt_centers.keys():
                    print(f"Optimizing pairs: tile {idx2} relative to fixed tile {idx1}")

                    # optimize pair of tiles, storing coordinates, cc matrix, and no. overlapping pixels
                    c2_opt, ccmat, n_overlap = optimize_pair(args['image_paths'], 
                                                             args['mask_paths'], 
                                                             tile_centers, 
                                                             idx1, 
                                                             idx2, 
                                                             args['max_shift'])
                    cc_matrices[(idx1,idx2)] = ccmat
                    opt_centers_all[(idx2,idx1)] = c2_opt
                    weights[(idx2,idx1)] = n_overlap

                    # compute optimized center as weighted mean of coordinates from all tile pairs
                    key_list = [key for key in opt_centers_all.keys() if key[0] == idx2]  
                    est_centers = np.array([opt_centers_all[key] for key in key_list])
                    est_weights = np.array([weights[key] for key in key_list]) 
                    est_weights /= np.sum(est_weights)
                    opt_centers[idx2] = np.sum(est_centers.T * est_weights, axis=1)
                    
    # save storage dicts
    for d,tag in zip([opt_centers, opt_centers_all, cc_matrices, weights], 
                     ['opt_centers', 'opt_centers_all', 'cc_matrices', 'weights']):
        handle = open(os.path.join(args['save_dir'], f"{tag}.pickle"), "wb")
        pickle.dump(d, handle)
        handle.close()
                    
    return opt_centers


#################################################
# Stitch all tiles from a particular tilt image #
#################################################

if __name__ == '__main__':

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)
    tile_centers, overlaps = set_up(args)
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    opt_centers = optimize_all(args, tile_centers, overlaps)

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

import numpy.ma as ma
import numpy as np
import mrcfile
import scipy
from scipy.spatial.distance import pdist
from scipy.signal import correlate2d
from scipy.optimize import minimize
from itertools import product, permutations
import time
import pathos.pools as pp
from pathos.multiprocessing import ProcessingPool

import matplotlib.pyplot as plt

'''
Zero-center normalize a tile or tiles (numpy nd array).

Input:
------
image: numpy array of numpy masked array.
mu (optional): mean pixel value.
std (optional): standard deviation of pixel value.

Output:
-------
normalized image such that it is zero-centerd and has standard deviation of 1.
'''
def preprocess(image, mu=None, std=None):
    if mu is None: mu = image.mean()
    if std is None: std = image.std()
    return (image - mu)/std
# ---- obsolete ----
# def preprocess(image):
#     return (image - image.mean())/image.std()

'''
Save numpy array in .mrc format. Credit to Ariana's code.

Input:
------
volume: numpy array, tile image.
savename: string, .mrc file path.
'''
def save_mrc(volume, savename):
    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(volume.astype(np.float32))
    mrc.close()
    return

'''
Get a tile in numpy array format. Note that only directory paths of tiles and
masks are taken as input, and the specific file of interest is assumed to have
naming convention "Xgrating2_{tilt_angle}_{tile_ind}.*" .

Input:
------
im_dir: string, path of directory where tile is stored as .mrc file.
tilt_angle: string/float/int, tilt angle of the tile of interest.
tile_ind: string/int, index of the tile of interest (0-indexed).

optionl:
masks_path: string, path of directory.
verbose: bool, print out min/max pixel value or not.
plot: bool, plot the masked tile or not.
plot_clim: tuple of two float, indicating the min/max pixel value for 
           visualization.

Output:
-------
tile: numpy 2d array of the tile of interest.
'''
def get_tile(im_dir, masks_path, tilt_angle, tile_ind, verbose=False, plot=True, plot_clim=(100,600)):
    mrc = mrcfile.open(im_dir+f'Xgrating2_{tilt_angle}_{tile_ind}.mrc')
    tile = mrc.data.copy()
    mrc.close()
    if verbose:
        print(tile.min())
        print(tile.max())
    if plot:
        mask = np.load(masks_path+f'Xgrating2_{tilt_angle}_{tile_ind}.npy', mmap_mode='r')
        plt.figure()
        plt.imshow(tile*mask, cmap='gray',origin='lower')
        plt.clim(plot_clim[0],plot_clim[1])
    return tile


'''
Stitch all tiles into a larger image

Input: (all other input parameter documentation see get_tile() above).
------
centers: 2d array of beam centers in unit of pixels, and the number of rows is
         the number of tiles to be stitched.
tile_shape: tuple of 2 int, shape of one tile 2d array.

Output:
-------
stitched: 2d numpy array, the stitched image.
'''
def stitch(centers, tile_shape, im_path, mask_dir, tilt_angle, tile_inds, verbose=True,
           plot=True, figure_size=(5,5), plot_clim=(100,600)):
    canvas_size = np.array(centers.max(axis=0)-centers.min(axis=0)+tile_shape, dtype=int)
    #1.2 is arbitrary to give some room of error/canvas margin
    canvas_size  = np.array(canvas_size*1.2, dtype=int) 
    if verbose: print(canvas_size)
    stitched = np.zeros(tuple(canvas_size))
    counts = np.zeros(tuple(canvas_size))
    
    #center the tiles (assuming already centered to (0,0)) to the center of the canvas
    if verbose: print(f'before centering canvas: {centers}')
    COM = np.array((canvas_size/2*centers.shape[0] -  centers.sum(axis=0))/centers.shape[0], 
                   dtype=int)
    centers += COM
    if verbose: print(f'after centering canvas: {centers}')
    
    Rx,Ry = tile_shape
    upleft = np.array(centers - [Rx/2, Ry/2],int)
    for i in range(centers.shape[0]):
        tile = get_tile(im_path, mask_dir, tilt_angle, tile_inds[i], plot=False)
        mask = np.load(mask_dir+f'Xgrating2_{tilt_angle}_{tile_inds[i]}.npy', mmap_mode='r')
        tile *= mask
        
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
#         print(f'{upleft_x} {upleft_y} {Rx} {Ry}')
#         print(act_im_mask.sum())
        assert(upleft_x >= 0 and upleft_y >= 0), f'tile out of canvas boundary! {upleft}'
#         #Approach 1. Overlapped area will be overwritten by later tiles
#         stitched[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry][act_im_mask] = images[i,:,:][act_im_mask]
        #Approach 2. Overlapped area pixels will be averaged rather than overwritten by later tiles
        stitched[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += tile
        counts[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry] += mask

    stitched[counts!=0] /= counts[counts!=0]

    if plot:
        plt.figure(figsize=figure_size)
        plt.imshow(stitched,cmap='gray',origin='lower')
        plt.clim(plot_clim[0],plot_clim[1])
#         plt.colorbar()
        plt.plot(centers[i,1], centers[i,0], 'bo', markersize=3)
    return stitched


'''
Return pairs of adjacent tile indices based on tile centers.

Input:
------
centers: numpy 2d array of (C,2) shape where C is the number of tiles , specify 
         tile centers.
cutoff: float, typically beam diameter. Tile centers within this distance is 
        considered a pair. 
        
Output:
-------
pairs: list of tuples, each tuple contains the indices of two tiles forming a pair.
'''
def get_pairs(centers, cutoff):
    pairs = list()
    dists = pdist(centers) < cutoff
    count = 0
    for i in range(centers.shape[0]):
        for j in range(i+1, centers.shape[0]):
            if dists[count]: pairs.append((i,j))
            count += 1
    assert(count == len(dists))
    return pairs

#reference function taken from Montage code to verify my cal_overlap
def cal_fractional_overlap(radius=1, stride=np.sqrt(3), n_overlaps=1):
    arc_angle = np.arccos(stride/2/radius) #in radians
    sector_area = (2*arc_angle * radius**2) / 2
    triangle_area = 2*np.sqrt(radius**2 - (stride/2)**2)*(stride/2)/2
    overlap_area = 2*(sector_area - triangle_area) #2 * half lemon shape
    f_overlap = n_overlaps * overlap_area / (np.pi*radius**2)
    return f_overlap

'''
Calculate properties of the overlapping portion between two tiles.

Input:
------
tile1/2: numpy 2d masked_array
center1/2: numpy array specifying tile centers. Note that center[0] is assumed
           to index row with origin='lower' -- that is vertial direction, but  
           larger value means higher up.
purpose: string, either 'CC' or 'fraction'. 
         If 'CC', this function will calculate the raw cross-correlation of the
         overlapping portion. If 'fraction', this function will only calculate
         the fraction of the overlapping region to the size of each tile (corner
         0-value pixels excluded).

Output:
-------
If purpose == 'fraction': return the overlap percentage for tile1 and tile 2.
If purpose == 'CC': return the raw cross-correlation value and the number of
                    pixels that contribute to the raw cc value.
'''
def cal_overlap(tile1, tile2, center1, center2, purpose='CC', verbose=False):
    assert(tile1.shape == tile2.shape), "assume same shape tiles"
    assert(purpose == 'CC' or purpose =='fraction')
    
    mask1 = tile1.mask
    mask2 = tile2.mask
    tile1 = tile1.data
    tile2 = tile2.data
    CC = 0
    n_pixels = 0
    m, n = tile1.shape 
    center1_r, center1_c = center1
    center2_r, center2_c = center2
    #number of overlapping rows and columns between the tiles
    nrows = int(m - np.abs(center2_r - center1_r))
    ncols = int(n - np.abs(center2_c - center1_c))
    if verbose: print(f'{nrows} {ncols}')
    if nrows <= 0 or ncols <= 0: return CC,n_pixels #pass tile boundary, no overlap
    if center2_r < center1_r:
        if center2_c < center1_c: 
            if purpose == 'CC': CC = tile2[-nrows:, -ncols:] * tile1[:nrows, :ncols]
            n_pixels = (1-mask2[-nrows:, -ncols:]) * (1-mask1[:nrows, :ncols])
            if verbose: print('tile 1 is at upper right of tile 2')
        else: 
            if purpose == 'CC': CC = tile2[-nrows:, :ncols] * tile1[:nrows, -ncols:]
            n_pixels = (1-mask2[-nrows:, :ncols]) * (1-mask1[:nrows, -ncols:])
            if verbose: print('tile 1 is at upper left of tile 2')
    else:
        if center2_c < center1_c: 
            if purpose == 'CC': CC = tile2[:nrows, -ncols:] * tile1[-nrows:, :ncols]
            n_pixels = (1-mask2[:nrows, -ncols:]) * (1-mask1[-nrows:, :ncols])
            if verbose: print('tile 1 is at lower right of tile 2')
        else:
            if purpose == 'CC': CC = tile2[:nrows, :ncols] * tile1[-nrows:, -ncols:]
            n_pixels = (1-mask2[:nrows, :ncols]) * (1-mask1[-nrows:, -ncols:])
            if verbose: print('tile 1 is at lower left of tile 2')
    
    n_pixels = np.array(n_pixels, dtype=int).sum()
#     non_zero_c = np.array((CC != 0), dtype=int).sum()
#     assert(n_pixels == non_zero_c),f'mask shows {n_pixels} but there are {non_zero_c} non-0 pixels.'
    
    if purpose == 'fraction': return [n_pixels/mask1.sum(), n_pixels/mask2.sum()]
    else: return [CC.sum(),n_pixels]


'''
Calculate cross-correlation matrix (similar to scipy.signal.correlate2d).

Input:
------
images_path: string, path to directory that stores .mrc tiles.
tilt_angle: string/float/int: tilt angle.
centers0_path: string, path to file of the initial guesses of the centers.
masks_path: string, path to diretory that stores all numpy 2d array masks.
pair: tuple of 2 ints, indices of the pair of tiles of interest
bound_p: int, search boundaries in one direction in unit of pixel size. 
         e.g. if bound_p = 1, it will search -1,0,1 for both x and y of
         the center of the second tile in the pair

Output:
-------
raw_ccmatrix:np array of shape (2*bound_p+1, 2*bound_p+1), where entry
             [i,j] is the raw cc value if the second tile center is 
             perturbed by adding [i-bound_p, j_bound_p]
num_pxmatrix: np array of same shape above, where each entry is the number
              of overlapping pixels for that perturbed stitch
'''
def get_CC_matrix(images_dir_path, tilt_angle, centers0_path, masks_path, pair, tile_inds, bound_p):
    tile1 = get_tile(images_dir_path, masks_path, tilt_angle, tile_inds[pair[0]], plot=False)
    tile2 = get_tile(images_dir_path, masks_path, tilt_angle, tile_inds[pair[1]], plot=False)
    mask1 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{tile_inds[pair[0]]}.npy',mmap_mode='r')
    mask2 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{tile_inds[pair[1]]}.npy',mmap_mode='r')
    
# def get_CC_matrix(images_dir_path, tilt_angle, centers0_path, masks_path, pair, bound_p):
#     tile1 = get_tile(images_dir_path, masks_path, tilt_angle, pair[0], plot=False)
#     tile2 = get_tile(images_dir_path, masks_path, tilt_angle, pair[1], plot=False)
#     mask1 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{pair[0]}.npy',mmap_mode='r')
#     mask2 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{pair[1]}.npy',mmap_mode='r')
    
    tile1 = ma.masked_array(tile1, 1-mask1)
    tile2 = ma.masked_array(tile2, 1-mask2)
    center1_fixed = np.load(centers0_path, mmap_mode='r')[pair[0],:]
    center2 = np.load(centers0_path, mmap_mode='r')[pair[1],:]
    
    raw_ccmatrix = np.ones((2*int(bound_p)+1, 2*int(bound_p)+1))
    num_pxmatrix = np.ones((2*int(bound_p)+1, 2*int(bound_p)+1))    
    for i in range(-int(bound_p),int(bound_p)+1):
        for j in range(-int(bound_p),int(bound_p)+1):
#             print(f'i={i} j={j}')
            raw_cc, npixels = cal_overlap(tile1, tile2, center1_fixed, center2+[i,j])
            raw_ccmatrix[i+int(bound_p), j+int(bound_p)] = raw_cc
            num_pxmatrix[i+int(bound_p), j+int(bound_p)] = npixels
    return raw_ccmatrix, num_pxmatrix

def __get_CC_matrix_wrapper__(args):
    return get_CC_matrix(*args)

'''
Greedy search optimized centers, where the search order follows the order of the
input centers array (stored in c_path).

Input: (all other input parameter documentation see get_CC_matrix() above).
------
n_processes: number of processors for parallelization (over pairs of tiles).
debug: bool, have intermediate print out messages or not.

Output: a tuple of three entries
--------------------------------
centers0: numpy array, optimized centers.
all_raw_ccmatrix (only in debug mode): lists of n_beam CC matrices. The shape of the ith
                                       CC matrix is (M, 2*bound_p+1, 2*bound_p+1), where
                                       M is the number of tiles the ith tile was optimized
                                       with respect to. Those M tiles must be optimized/
                                       fixed already and overlapping with the ith tile.
all_num_pxmatrix (only in debug mode): lists of n_beam matrices of number of overlapping
                                       pixels.
all_displacements (only in debug mode): lists of n_beam arrays. The shape of the ith
                                        entry is (M, 2), indicating the optimal displacements
                                        for the ith beam with respect to those M fixed tiles.
'''
def opt_CC_greedy(im_path, pairs_path, c_path, m_path, bound_p, tilt_angle, tile_inds,
                  n_processes=4, debug=False):

    centers0 = np.load(c_path).copy()
    pairs = np.load(pairs_path, mmap_mode='r')

    all_raw_ccmatrix = []
    all_num_pxmatrix = []
    all_displacements = []
    for i in range(1, len(tile_inds)): #assume first center is already fixed
#     for i in range(1, centers0.shape[0]): #assume first center is already fixed
        rel_pairs = [p for p in pairs if i == p[1]]
#         if debug: print(f'greedy at tile {i} with relevant pairs {rel_pairs}')
        rel_pairs_act_ind = [(tile_inds[p[0]], tile_inds[p[1]]) for p in rel_pairs]
        if debug: print(f'greedy at tile {tile_inds[i]} with relevant pairs {rel_pairs_act_ind}')
        
        pool = pp.ProcessPool(n_processes)
        all_cc_npix = pool.map(__get_CC_matrix_wrapper__, 
                           zip([im_path]*len(rel_pairs),
                               [tilt_angle]*len(rel_pairs),
                               [c_path]*len(rel_pairs),
                               [m_path]*len(rel_pairs),
                               rel_pairs,
                               [tile_inds]*len(rel_pairs),
                               [bound_p]*len(rel_pairs)))
        
        raw_ccmatrix = [x[0] for x in all_cc_npix]
        num_pxmatrix = [x[1] for x in all_cc_npix]
        all_raw_ccmatrix.append(raw_ccmatrix)
        all_num_pxmatrix.append(num_pxmatrix)
        
        norm_cc = np.array(raw_ccmatrix)/np.array(num_pxmatrix)
        #Approach 1. Find optimal displacements of this center in each
        #involved pair, then take the average displacement.
        mat_ijs = np.array([np.unravel_index(np.argmax(norm), norm.shape) for norm in norm_cc])
        mat_i, mat_j = np.round(mat_ijs.mean(axis=0))
        if debug: 
#             print(f'center {i} mat_ijs={mat_ijs}')
            print(f'center {i} (tile {tile_inds[i]}) mat_ijs={mat_ijs}')
            print(f'mean perturbed mat_i={mat_i} mat_j={mat_j}')
#Approach 2. Average the CC matrices first then find the optimal displacement
#         norm_cc = raw_ccmatrix.sum(axis=0)/num_pxmatrix.sum(axis=0)
#         mat_i, mat_j = np.unravel_index(np.argmax(norm_cc), raw_ccmatrix.shape[1:])
        all_displacements.append(mat_ijs-bound_p)
        centers0[i,:] += np.array([mat_i-bound_p,mat_j-bound_p], dtype=int)
        np.save(c_path, centers0)

    if debug: return (centers0, all_raw_ccmatrix, all_num_pxmatrix, all_displacements)
    return (centers0,-1,-1,-1)


#=================Functions below were not outdated or not tested=====================#
# def get_ImageShift(tilt_angle, tile_ind, pixel_unit=True):
#     pixel_size = 1
#     with open(f'Xgrating2_{tilt_angle}_{tile_ind}.mrc.mdoc', 'r') as f_mrc:
#         for line in f_mrc:
#             if 'PixelSpacing' in line and pixel_unit:
#                 pixel_size = float(line.split('=')[1])
#             if 'ImageShift' in line: 
#                 im_shift = np.array(line.split('=')[1].split(), dtype=float)
#                 print(f'ImageShift (um): {im_shift}')
#                 im_shift = np.around(im_shift*1e4/pixel_size)
#                 im_shift = np.array(im_shift, dtype=int)
#                 print(f'ImageShift (pixels): {im_shift}')
#                 return im_shift

    
# '''
# Calculate the total normalized cross-correlation score of the entire
# image/canvas for one tilt angle.

# Input:
# ------
# images_path: string, path to file that stores all numpy nd array tiles.
# pairs_path: string, path to file that stores the list of pairs.
# centers0_path: string, path to file that stores initial guesses of the
#                centers.
# perturbation: 1d list of float of length 2*C, where C is the number of
#               tiles. Essentially a flattened array that stores the
#               perturbation to x and y coordinate of each tile center.

# Output:
# -------
# float, total cross correlation normalized by total number of overlapping
# pixels.
# '''
# def cal_total_CC(images_dir_path, pairs_path, centers0_path, masks_path, perturbation):
#     images = np.load(images_dir_path, mmap_mode='r')
#     masks = np.load(masks_path, mmap_mode='r')
    
#     tilt_angle = 0
#     pairs = np.load(pairs_path, mmap_mode='r')
#     centers0 = np.load(centers0_path, mmap_mode='r').copy()
#     centers0[1:,] += np.array(perturbation).reshape(-1,2)
#     raw_CC_total = 0
#     npixel_total = 0
#     for p in pairs:
#         i,j = p
#         tile1 = ma.masked_array(images[i,:,:],1-masks[i,:,:])
#         tile2 = ma.masked_array(images[j,:,:],1-masks[j,:,:])
# #         tile1_path = images_dir_path+f'Xgrating2_{tilt_angle}_{i}.npy'
# #         tile2_path = images_dir_path+f'Xgrating2_{tilt_angle}_{j}.npy'
# #         tile1 = np.load(tile1_path,mmap_mode='r')
# #         tile2 = np.load(tile2_path,mmap_mode='r')
# #         mask1 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{i}.npy',mmap_mode='r')
# #         mask2 = np.load(masks_path+f'Xgrating2_{tilt_angle}_{j}.npy',mmap_mode='r')
# #         tile1 = ma.masked_array(tile1,1-mask1)
# #         tile2 = ma.masked_array(tile2,1-mask2)
        
#         center1 = centers0[i,:]
#         center2 = centers0[j,:]
#         raw_cc, npixel = cal_overlap(tile1, tile2, center1, center2)
#         raw_CC_total += raw_cc
#         npixel_total += npixel
#     print(f'cc={raw_CC_total} nn={npixel_total}')    
#     return raw_CC_total/npixel_total

# def __cal_total_CC_wrapper__(args):
#     return cal_total_CC(*args)

# '''
# Brute force search what centers give highest total CC score.
# Not updated - do NOT use. 

# Output:
# -------
# centers0: numpy array, optimized centers.
# '''
# def opt_CC_brute_force(im_path, pr_path, cent_path, bound_p, 
#                        verbose=True, n_processes=4):
#     centers0 = np.load(cent_path, mmap_mode='r').copy()
#     perturbation = product(np.arange(-int(bound_p),int(bound_p)+1),
#                            repeat=centers0.flatten().shape[0]-2)
#     num_products = (2*int(bound_p)+1)**(centers0.flatten().shape[0]-2)
#     pool = pp.ProcessPool(n_processes)
#     all_norm_cc = pool.map(__cal_total_CC_wrapper__, 
#                            zip([im_path]*num_products,
#                                [pr_path]*num_products,
#                                [c_path]*num_products,
#                                perturbation))
#     all_norm_cc = np.array(all_norm_cc)
    
    
#     perturbation = product(np.arange(-int(bound_p),int(bound_p)+1),
#                            repeat=centers0.flatten().shape[0]-2)
#     opt_pert = np.array(list(perturbation))[all_norm_cc.argmax()]
#     centers0[1:,:] += opt_pert
#     return centers0  
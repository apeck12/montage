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
Zero-center normalize the image (numpy nd array)
'''
def preprocess(image):
    return (image - image.mean())/image.std()

'''
Stitch all tiles into a larger image

Input:
------
ma_tiles: numpy masked array of shape (C,R,R) where C is the number of tiles.
centers: numpy array of shape (C, 2), optimized centers of each tile.
canvas_diag_corners: numpy array of shape (2,2), where the first row index the
                     upper left corner pixel and the second row index the bottom
                     right corner pixel. Typicall is [(0,0), canvas.shape].
plot: bool, whether plot the stitched image.
figure_size: tuple of int, only valid if plot is True.

Output:
-------
stitched: 2d numpy array, the stitched image.
'''
def stitch(ma_tiles, centers, canvas_diag_corners, plot=True, figure_size=(5,5)):
    images = ma_tiles.data
    masks = ma_tiles.mask
    stitched = np.zeros(canvas_diag_corners[1,:]-canvas_diag_corners[0,:])
    centers -= canvas_diag_corners[0,:]
    N,Rx,Ry = images.shape #Assume beam is at the center of the tile
    upleft = np.array(centers - Rx/2,int)
    for i in range(centers.shape[0]):
        upleft_x, upleft_y = np.array(upleft[i,:], dtype=int)
        act_im_mask = np.array(1-masks[i,:,:], dtype=bool)
#         act_im_mask = images[i,:,:] != 0
#         print(f'{upleft_x} {upleft_y} {Rx} {Ry}')
        assert(upleft_x >= 0 or upleft_y >= 0), 'tile out of canvas boundary!'
        #Overlapped area will be overwritten by later tiles
        stitched[upleft_x:upleft_x+Rx, upleft_y:upleft_y+Ry][act_im_mask] = images[i,:,:][act_im_mask]

    if plot:
        plt.figure(figsize=figure_size)
        plt.imshow(stitched,cmap='gray')
        plt.clim(-2,2)
#         plt.colorbar()
        plt.plot(centers[:,1], centers[:,0], 'bo', markersize=3)
    return stitched


'''
Return pairs of adjacent tile indices based on tile centers

Input:
------
centers: numpy 2d array of (C,2) shape where C is the number of tiles , specify tile centers
cutoff: float, typically beam diameter. Tile centers within this distance is considered
        a pair. 
        
Output:
-------
pairs: list of tuples, each tuple contains the indices of two tiles forming a pair
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
TODO: need to change to accept masks as additional parameters!!!

Input:
------
tile1/2: numpy 2d array
center1/2: numpy array specifying tile centers
purpose: string, either 'CC' or 'fraction'. 
         If 'CC', this function will calculate the raw cross-correlation of the
         overlapping portion. If 'fraction', this function will only calculate
         the fraction of the overlapping region to the size of each tile (corner
         0-value pixels excluded).

Output:
-------
If purpose == 'fraction': return the overlap percentage for tile1 and tile 2 
                          (should be the same if beam shape is the same).
If purpose == 'CC': return the raw cross-correlation value and the number of
                    pixels that contribute to the raw cc value.
'''
def cal_overlap(tile1, tile2, center1, center2, purpose='CC'):
    assert(tile1.shape == tile2.shape), "assume same shape tiles"
    assert(purpose == 'CC' or purpose =='fraction')
    
    CC = 0
    n_pixels = 0
    m, n = tile1.shape 
    center1_r, center1_c = center1
    center2_r, center2_c = center2
    #number of overlapping rows and columns between the tiles
    nrows = int(m - np.abs(center2_r - center1_r))
    ncols = int(n - np.abs(center2_c - center1_c))
    if nrows <= 0 or ncols <= 0: 
#             print(f'center1_r:{center1_r} center1_c:{center1_c} center2_r:{center2_r} center2_c:{center2_c}')
#             print(f'm:{m} n:{n}')
#             print(f'nrows:{nrows} ncols: {ncols}')
        return CC,n_pixels #pass tile boundary, no overlap
    if center2_r < center1_r:
        if center2_c < center1_c:
            CC = tile2[-nrows:, -ncols:] * tile1[:nrows, :ncols]
        else: CC = tile2[-nrows:, :ncols] * tile1[:nrows, -ncols:]
    else:
        if center2_c < center1_c:
            CC = tile2[:nrows, -ncols:] * tile1[-nrows:, :ncols]
        else: CC = tile2[:nrows, :ncols] * tile1[-nrows:, -ncols:]
    n_pixels = (CC != 0).sum()
    
    if purpose == 'fraction': return [n_pixels/tile1.sum(), n_pixels/tile2.sum()]
    else: return [CC.sum(),n_pixels]
    
    
'''
Calculate the total normalized cross-correlation score of the entire
image/canvas for one tilt angle.

Input:
------
images_path: string, path to file that stores all numpy nd array tiles.
pairs_path: string, path to file that stores the list of pairs.
centers0_path: string, path to file that stores initial guesses of the
               centers.
perturbation: 1d list of float of length 2*C, where C is the number of
              tiles. Essentially a flattened array that stores the
              perturbation to x and y coordinate of each tile center.

Output:
-------
float, total cross correlation normalized by total number of overlapping
pixels.
'''
def cal_total_CC(images_path, pairs_path, centers0_path, perturbation):
    images = np.load(images_path, mmap_mode='r')
    pairs = np.load(pairs_path, mmap_mode='r')
    centers0 = np.load(centers0_path, mmap_mode='r').copy()
    centers0[1:,] += np.array(perturbation).reshape(-1,2)
    raw_CC_total = 0
    npixel_total = 0
    for p in pairs:
        i,j = p
        tile1 = images[i,:,:]
        tile2 = images[j,:,:]
        center1 = centers0[i,:]
        center2 = centers0[j,:]
        raw_cc, npixel = cal_overlap(tile1, tile2, center1, center2)
        raw_CC_total += raw_cc
        npixel_total += npixel
    print(f'cc={raw_CC_total} nn={npixel_total}')    
    return raw_CC_total/npixel_total

def __cal_total_CC_wrapper__(args):
    return cal_total_CC(*args)

'''
Brute force search what centers give highest total CC score.
Not updated - do NOT use. 

Output:
-------
centers0: numpy array, optimized centers.
'''
def opt_CC_brute_force(im_path, pr_path, cent_path, bound_p, 
                       verbose=True, n_processes=4):
    centers0 = np.load(cent_path, mmap_mode='r').copy()
    perturbation = product(np.arange(-int(bound_p),int(bound_p)+1),
                           repeat=centers0.flatten().shape[0]-2)
    num_products = (2*int(bound_p)+1)**(centers0.flatten().shape[0]-2)
    pool = pp.ProcessPool(n_processes)
    all_norm_cc = pool.map(__cal_total_CC_wrapper__, 
                           zip([im_path]*num_products,
                               [pr_path]*num_products,
                               [c_path]*num_products,
                               perturbation))
    all_norm_cc = np.array(all_norm_cc)
    
    
    perturbation = product(np.arange(-int(bound_p),int(bound_p)+1),
                           repeat=centers0.flatten().shape[0]-2)
    opt_pert = np.array(list(perturbation))[all_norm_cc.argmax()]
    centers0[1:,:] += opt_pert
    return centers0  

'''
Calculate cross-correlation matrix (similar to scipy.signal.correlate2d).

Input:
------
images_path: string, path to file that stores all numpy nd array tiles.
centers0_path: string, path to file that stores initial guesses of the
               centers.
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
def get_CC_matrix(images_path, centers0_path, pair, bound_p):
    tile1 = np.load(images_path, mmap_mode='r')[pair[0],:,:]
    tile2 = np.load(images_path, mmap_mode='r')[pair[1],:,:]
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

Output: a tuple of three entries
--------------------------------
centers0: numpy array, optimized centers.
all_raw_ccmatrix (only in debug mode): lists of n_beam CC matrices.
all_num_pxmatrix (only in debug mode): lists of n_beam matrices of number of overlapping
                                       pixels. 
'''
def opt_CC_greedy(im_path, pairs_path, c_path, bound_p, n_processes=4, debug=False):

    centers0 = np.load(c_path).copy()
    pairs = np.load(pairs_path, mmap_mode='r')

    all_raw_ccmatrix = []
    all_num_pxmatrix = []
    for ind in range(1, centers0.shape[0]): #assume first center is already fixed
        rel_pairs = [p for p in pairs if ind == p[1]]
        if debug: print(f'greedy at tile {ind} with relevant pairs {rel_pairs}')
        
        pool = pp.ProcessPool(n_processes)
        all_cc_npix = pool.map(__get_CC_matrix_wrapper__, 
                           zip([im_path]*len(rel_pairs),
                               [c_path]*len(rel_pairs),
                               rel_pairs,
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
            print(f'center {ind} mat_ijs={mat_ijs}')
            print(f'mean perturbed mat_i={mat_i} mat_j={mat_j}')
#Approach 2. Average the CC matrices first then find the optimal displacement
#         norm_cc = raw_ccmatrix.sum(axis=0)/num_pxmatrix.sum(axis=0)
#         mat_i, mat_j = np.unravel_index(np.argmax(norm_cc), raw_ccmatrix.shape[1:])
        centers0[ind,:] += [mat_i-bound_p,mat_j-bound_p]
        np.save(c_path, centers0)

    if debug: return (centers0, all_raw_ccmatrix, all_num_pxmatrix)
    return (centers0,-1,-1)
import argparse, time, os, mrcfile, glob
import scipy.optimize, skimage.filters
from natsort import natsorted
import numpy as np
import utils

"""
Determine the Fresnel-contaminated region based on a heuristic approach involving
bandpass filtering and generate a mask to eliminate the Fresnel-contaminated and
unilluminated region of each projection image.
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
    parser.add_argument('-ls','--low_sigma', help='Sigma value for lower kernel of bandpass filter',
                        required=False, type=float, default=2)
    parser.add_argument('-hs','--high_sigma', help='Sigma value for upper kernel of bandpass filter',
                        required=False, type=float, default=3)
    parser.add_argument('-f','--fraction', help='Fraction of max. filtered intensity to estimate fringe ring',
                        required=False, type=float, default=0.4)
    parser.add_argument('-b','--buffer', help='No. pixels buffer for a more conservative fringe removal',
                         required=False, type=int, default=2)
    parser.add_argument('-o','--save_dir', help='Directory to which to save output files',
                        required=True, type=str)

    return vars(parser.parse_args())


###############################################################
# Functions for heuristic approach to masking Fresnel fringes #
###############################################################

def cart2pol(shape, center):
    """
    Compute polar coordinates for a 2d grid of specified dimensions.
    
    Inputs:
    -------
    shape: tuple of 2d grid shape
    center: tuple of (x,y) positions of circle's center
    
    Outputs:
    --------
    rho: 2d array specifying radius of each pixel
    phi: 2d array specifying polar angle of each pixel, in degrees
    """
    y, x = np.indices(shape).astype(float)
    x -= center[0] 
    y -= center[1] 
    rho = np.sqrt(np.square(x) + np.square(y))
    phi = np.rad2deg(np.arctan2(y, x))
    return(rho, phi)


def fresnel_mask(image_path, low_sigma=2, high_sigma=3, fraction=0.4, buffer=2):
    """
    Generate a Fresnel mask by bandpass-filtering the tile, followed by fitting a circle
    to the resulting ring of high-intensity pixels at the edge of the illuminated region.
    NB: recommendations of low_sigma=2, high_sigma=3 for cellular images, 8x-binned.
    
    Inputs:
    -------
    image_path: path to tile in MRC format
    low_sigma: standard deviation of Gaussian kernel with smaller sigma for filter
    high_sigma: standard deviation of Gaussian kernel with larger sigma for filter
    fraction: fraction of filtered intensity maximum for thresholding pixels in ring
    
    Outputs:
    --------
    mask: 2d array, where 1 corresponds to illuminated, fringe-uncontaminated pixels
    """
    def fit_circle(params):
        """
        Cost function for fitting a circle to points; parameters are center and radius.
        """
        xc, yc, r = params
        r_p = np.sqrt((x-xc)**2 + (y-yc)**2)
        return r - r_p

    # load tile and apply bandpass filter
    tile = mrcfile.open(image_path).data.copy()
    tile = skimage.filters.difference_of_gaussians(tile, low_sigma, high_sigma)
    x,y = np.where(tile>=fraction*tile.max())
    xi,yi,ri = tile.shape[0]/2, tile.shape[1]/2, tile.shape[0]/2
    
    # remove points within 0.9x unoptimized radius to focus on edge region
    rho_i, phi_i = cart2pol(tile.shape, (yi,xi))
    r_mask = np.zeros_like(tile)
    r_mask[rho_i<0.9*ri] = 1
    r_threshold = r_mask[x,y]
    x,y = x[r_threshold==0], y[r_threshold==0]

    # fit circle to points, assuming sufficient points
    if len(x) > 3:
        p_opt, ier = scipy.optimize.leastsq(fit_circle, (xi,yi,ri))
    else:
        p_opt = (xi,yi,ri) 

    # mask any points outside radius with small buffer
    rho,phi = cart2pol(tile.shape, (p_opt[1],p_opt[0]))
    mask = np.ones_like(tile)
    mask[np.where(rho>p_opt[2]-buffer)] = 0
    
    return mask


######################################################
# Generate a binary mask for each image in path list #
######################################################

if __name__ == '__main__':

    start_time = time.time()

    # set up
    args = parse_commandline()
    args['image_paths'] = natsorted(glob.glob(args['image_paths']))
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    # generate masks
    for i,fname in enumerate(args['image_paths']):
        print(f"Masking image no. {i} of {len(args['image_paths'])}")
        nname = fname.split('/')[-1] 
        outpath = os.path.join(args['save_dir'], nname)
        mask = fresnel_mask(fname, low_sigma=args['low_sigma'], high_sigma=args['high_sigma'], 
                            fraction=args['fraction'], buffer=args['buffer'])
        utils.save_mrc(mask, outpath)

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

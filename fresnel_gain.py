import scipy.optimize, skimage.filters
from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse
import mrcfile, itertools, utils
import numpy as np

"""
Generate a mask in the shape of each tile for removal of Frensel fringes and the
unilluminated region (value of 0) and optionally application of gain correction.
Also save the optimized beam fit parameters: (xc, yc, radius) for each tile.
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
                        required=False, type=float, default=0.2)
    parser.add_argument('-gc','--gain_correct', help='Whether to apply a gain correction',
                        action='store_true') # defaults to False if argument is not supplied
    parser.add_argument('-o','--save_dir', help='Directory to which to save output files',
                        required=True, type=str)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    # expand paths keys into ordered lists
    args['image_paths'] = natsorted(glob.glob(args['image_paths']))
    
    # create output directory if doesn't already exist
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])
    
    return args


#######################################################
# Useful functions for dealing with circular profiles #
#######################################################

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


def radial_profile(radii,intensities):
    """
    Efficient method to compute an average radial profile for a 2d image. Code is modified from: 
    https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile.
    Note that bins without counts are disregarded, so output rx may not start at 0.
    
    Inputs:
    -------
    radii: np.array of radius values
    intensities: np.array of intensities, corresponding to rho
    
    Outputs:
    --------
    rx: radius of each bin present in rprofile
    rprofile: mean radial profile of image
    """
    r = radii.copy().astype(np.int)
    rx = np.unique(r.ravel())
    tbin = np.bincount(r.ravel(), intensities.ravel())
    nr = np.bincount(r.ravel())
    rprofile = tbin[nr!=0] / nr[nr!=0]
    return rx, rprofile 


#################################################
# Fresnel-masking and gain correction functions #
#################################################

def compute_fresnel_edge(image_path, low_sigma=2, high_sigma=3, fraction=0.4):
    """
    Compute the start of the Fresnel region by bandpass-filtering the tile, followed by 
    fitting a circle to the resulting ring of high-intensity pixels at the edge of the 
    illuminated region and determining the inner radius of this ring.
    
    Inputs:
    -------
    image_path: path to tile in MRC format
    low_sigma: standard deviation of Gaussian kernel with smaller sigma for filter
    high_sigma: standard deviation of Gaussian kernel with larger sigma for filter
    fraction: fraction of filtered intensity maximum for thresholding pixels in ring
    
    Outputs:
    --------
    p_opt: tuple of optimized beam parameters, (h,k,r)
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
        print("Warning: insufficient points, fitting Fresnel region to 0.95*radius")
        p_opt = (xi,yi,0.95*ri) 
        
    # compute radial profile using updated beam parameters
    rho_i, phi_i = cart2pol(tile.shape, (p_opt[1], p_opt[0]))
    rx_i, rp_i = radial_profile(rho_i.flatten(), tile.flatten())
    rp_i[rx_i<0.1*p_opt[2]] = 0.05*rp_i[rx_i<0.1*p_opt[2]] # eliminate peaks near center

    # select the radial indices that span the tile's edge, including fringe region
    max_idx = rx_i[rp_i==rp_i.max()][0]
    min_idx = int(max_idx - 0.05*p_opt[2])

    # determine inner radius of ring as fraction of peak's max intensity
    sel = rp_i[min_idx:max_idx]
    nsel = (sel - sel.min()) / (sel.max() - sel.min())
    dsel = np.abs(nsel - fraction*nsel.max())
    p_opt[2] = rx_i[min_idx:max_idx][dsel==dsel.min()][0]

    return p_opt


def compute_gain_reference(opt_beam, image_paths, kernel_size=None):
    """
    Compute a gain reference, assuming that the gain is characterized by a gradual
    reduction in radial intensity. The reference is computed from the mean radial 
    intensity profiles of all tiles for a particular tilt angle.
    
    Inputs:
    -------
    opt_beam: dict of optimized beam parameters (h,k,r) for each tile
    image_paths: list of tile paths
    kernel_size: size of kernel for median filter, optional
    
    Outputs:
    --------
    gain_x: radii (in pixels) associated with gain_y values
    gain_y: median-filtered average gain correction profile
    rx: dict of radii for radial profiles
    rp: dict of intensities for radial profiles
    """
    # compute radial profiles of all tiles
    avg_radius = np.mean(np.array(list(opt_beam.values()))[:,2])
    rx, rp = OrderedDict(), OrderedDict()

    for xi,ipath in enumerate(image_paths):
        tile = mrcfile.open(ipath).data
        rho,phi = cart2pol(tile.shape, (opt_beam[xi][1], opt_beam[xi][0]))
        rx_i, rp_i = radial_profile(rho.flatten(), tile.flatten())
        rp_i /= np.mean(rp_i[int(0.1*avg_radius)]) # normalize 
        rx[xi], rp[xi] = rx_i, rp_i
    c_rx, c_rp = np.hstack(list(rx.values())), np.hstack(list(rp.values()))

    # re-order values for increasing radius and median filter intensity profile
    sorted_idx = np.argsort(c_rx)
    c_rx, c_rp = c_rx[sorted_idx], c_rp[sorted_idx]
    start_idx = np.where(c_rx>0.1*avg_radius)[0][0]
    if kernel_size is None:
        kernel_size = int(avg_radius / 10.0)
        if kernel_size % 2 == 0: kernel_size += 1
    medfilt = scipy.signal.medfilt(c_rp, kernel_size)

    # compute gain profile, padding on either side
    gain_x, gain_y = c_rx[start_idx:], medfilt[start_idx:]
    add_length0, add_length1 = gain_x[0], int(gain_x.max()) + 1
    gain_x = np.concatenate((np.arange(0, add_length0), gain_x, np.arange(add_length1,add_length1+100)))
    gain_y = np.concatenate((np.ones(int(add_length0)), gain_y, np.zeros(100)))
    gain_y[gain_y>0.5] = 1.0 / gain_y[gain_y>0.5] # invert reasonable values

    return gain_x, gain_y, rx, rp


###############################################
# Generate a mask for each image in path list #
###############################################

if __name__ == '__main__':

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    # fit beam and start of fresnel region
    opt_params = OrderedDict()
    for xi,image_path in enumerate(args['image_paths']):
        opt_params[xi] = compute_fresnel_edge(image_path,
                                              low_sigma=args['low_sigma'],
                                              high_sigma=args['high_sigma'], 
                                              fraction=args['fraction'])

    # determine gain correction based on optimized beam parameters
    if args['gain_correct'] is True:
        gain_x, gain_y, rx, rp = compute_gain_reference(opt_params, args['image_paths'])
        f = scipy.interpolate.interp1d(gain_x, gain_y)

    # generate fresnel mask for each tile, optionally with gain correction
    mshape = mrcfile.open(args['image_paths'][0]).data.shape
    for xi,ipath in enumerate(args['image_paths']):
    
        print(f"Masking image no. {xi} of {len(args['image_paths'])}")
        mask = np.ones(mshape).astype(float)
        rho,phi = cart2pol(mshape, (opt_params[xi][1], opt_params[xi][0]))
    
        if args['gain_correct'] is True:
            gain_y_interp = f(rho)
            gain_y_interp[rho>opt_params[xi][2]] = 0
            mask *= gain_y_interp
    
        else:
            mask[rho>opt_params[xi][2]] = 0
    
        outpath = os.path.join(args['save_dir'], ipath.split('/')[-1] )
        utils.save_mrc(mask, outpath)

    # save the beam fit parameters, ordered by tile index
    np.save(os.path.join(args['save_dir'], "params.npy"), np.array(list(opt_params.values())))
    
    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")

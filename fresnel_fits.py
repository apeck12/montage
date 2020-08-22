from natsort import natsorted
import glob, time, os
import numpy as np
import mrcfile, argparse
import pathos.pools as pp

from scipy.signal import argrelextrema
import scipy.signal, scipy.interpolate
import scipy.optimize

"""
Generate a series of masks for input projection images, where a value of 1 corresponds to the 
Fresnel-unaffected, illuminated region by the beam. Masks are saved as npy files with the same
prefix as the projection image from which they were generated.
"""


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Generate masks for unillumninated and Fresnel regions.')
    parser.add_argument('-i','--input', help='Path to filenames in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-o','--output', help='Directory to whcih to save output masks', 
                        required=True, type=str)
    parser.add_argument('-n','--n_processes', help='Number of CPU processors over which to distrubte task',
                        required=True, type=int)
    parser.add_argument('-s','--sigma', help='Contrast sigma threshold used to determine Fresnel boundary',
                        required=False, default=4.0, type=float)
    parser.add_argument('-ai','--interval', help='Angular increment in degrees for estimating Fresnel radius',
                        required=False, default=4.0, type=float)
    parser.add_argument('-b','--b_mask', help='Intensity threshold for generating rough beam mask',
                        required=False, default=100.0, type=float)
    parser.add_argument('-ss','--step_size', help='Angular step between overlapping wedges and for interpolation',
                        required=False, default=0.2, type=float)
    parser.add_argument('-w','--window', help='Length of window for Wiener filtering, must be odd integer',
                        required=False, default=35, type=int)

    return vars(parser.parse_args())


################################################
# Generic functions useful for Fresnel-fitting #
################################################

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


def mask_circle(shape,h,k,r):
    """
    Generate a 2d array of input shape, where pixels that lie within the circle of radius r and
    center (h,k) are set to have a value of 1.
    
    Inputs:
    -------
    shape: (x,y) dimensions of 2d array to generate
    h,k: x,y positions of circle's center
    r: radius of circle
    
    Outputs:
    --------
    cmask: 2d array with pixels in specified circle given a value of 1
    """
    xx, yy = np.meshgrid(range(shape[1]), range(shape[0]))
    radii = np.sqrt(np.square(xx - h) + np.square(yy - k))
    cmask = np.zeros(shape)
    cmask[radii<r] = 1
    return cmask


def fit_beam(mimage, x0=None):
    """
    Fit a circle to the illuminated area of mimage.
    
    Inputs:
    -------
    mimage: thresholded beam image, with approximate illuminated area set to 1
    x0: array of initial guess for [h,k,r] parameters
    
    Outputs:
    --------
    results: optimized [h,k,r] parameters
    """

    def objective(params):
        h,k,r = params
        gimage = mask_circle(mimage.shape,h,k,r)
        return -1*np.corrcoef(mimage.flatten(), gimage.flatten())[0,1]
    
    if x0 is None:
        x0 = [mimage.shape[0]/2, mimage.shape[1]/2, 0.42*mimage.shape[1]]
    res = scipy.optimize.minimize(objective, x0, method = 'Nelder-Mead')
    
    return res.x


def local_min_scipy(arr):
    """
    Find indices of all local minima of 1d array arr.
    """
    minima = argrelextrema(arr, np.less_equal)[0]
    return minima


def local_max_scipy(arr):
    """
    Find indices of all local maxima of 1d array arr.
    """
    minima = argrelextrema(arr, np.greater_equal)[0]
    return minima


#################################################
# Functions for fitting Fresnel-affected region #
#################################################

def fresnel_radius(r_beam, r_vals, d_vals, sigma=4):
    """
    Estimate the radius at which the Fresnel fringes begin, based on when the contrast
    from the Fresnel fringes exceeds sigma times the mean signal's contrast. Contrast
    is local and computed as (I_max - I_min) / I_min, where I_max and I_min are adjacent 
    extrema along the smoothed radial profile. At Fresnel fringes, the contrast will be 
    the intensity difference between adjacent crests and troughs.
    
    Inputs:
    -------
    r_beam: estimated radius of beam
    r_vals: 1d array of radius values
    d_vals: 1d array of corresponding intensity values
    sigma: threshold for Fresnel region, relative to signal contrast
    
    Output:
    -------
    r_fresnel: estimated radius at which Fresnel fringes begin
    r_edge: updated estimated radius of beam based on fall-off in signal for wedge
    """
    from scipy.signal import savgol_filter
    
    sr_start, sr_end = 0.4, 0.7 # signal region start and end, fraction of radius
    fr_start = 0.95 # suspected fresnel region start, fraction of radius
    
    # compute average radial profile and filter; compute extrema
    rx, rprofile = radial_profile(r_vals, d_vals)
    rp_filt = savgol_filter(rprofile.copy(), window_length=7, polyorder=3)
    max_idx, min_idx = local_max_scipy(rp_filt), local_min_scipy(rp_filt)
    
    # alter min_idx and max_idx such that first index will be a minimum
    gap = np.abs(len(max_idx) - len(min_idx))
    if len(max_idx) != len(min_idx):
        if len(max_idx) > len(min_idx):
            max_idx = max_idx[gap:]
        else:
            if min_idx[0] < max_idx[0]: 
                min_idx = min_idx[:-1*gap]
            else:
                max_idx, min_idx = max_idx[1:], min_idx[:-(gap+1)]
    rx_max_idx, rx_min_idx = rx[max_idx], rx[min_idx] # indices in terms of radius
    
    # compute local contrast based on intensity variation between adjacent extrema
    contrast = (rp_filt[max_idx] - rp_filt[min_idx]) / rp_filt[min_idx]
    
    # estimate contrast for signal region and start of fresnel region
    c_signal = np.mean(contrast[np.where((rx_max_idx > sr_start*r_beam) & (rx_max_idx < sr_end*r_beam))])
    r_edge = rx_max_idx[rx_max_idx < r_beam][-1]
    
    try:
        start_idx1 = np.where((rx_max_idx > fr_start*r_beam) & (contrast > sigma*c_signal))[0][0]
        start_idx2 = np.where((rx_min_idx > fr_start*r_beam) & (contrast > sigma*c_signal))[0][0]
        r_fresnel = min(rx_max_idx[start_idx1], rx_min_idx[start_idx2])
    except:
        r_fresnel = r_edge
    if r_fresnel > r_beam: r_fresnel = r_edge

    return r_fresnel, r_edge


def fit_by_wedge(data, rho, phi, r_f, offset, interval=3.0, sigma=4):
    """
    Estimate Fresnel region for full image by dividing it into a series of wedges, each
    spanning an angular width given by interval. 
    
    Inputs:
    -------
    data: projection image
    rho, phi: arrays corresponding to polar coordinates of each pixel in data
    r_f: estimated radius of beam 
    offset: starting angular increment of first wedge
    interval: angular width of each wedge in degrees, optional
    sigma: contrast threshold used by fresnel_radius function, optional
    
    Outputs:
    --------
    phi_vals: array of phi values at which Fresnel radius was estimated
    rf_vals: array of estimated radii at which fringes at associated phi_vals
    re_vals: array of estimated outer radii of beam at associated phi_vals
    mask: mask with a value of 1 for illuminated pixels unaffected by fringes
    """
    
    # set up angular search and storage arrays
    mask = np.zeros_like(data)
    angle_bins = np.arange(offset, 360+offset, interval)
    phi_vals, rf_vals, re_vals = np.zeros_like(angle_bins), np.zeros_like(angle_bins), np.zeros_like(angle_bins)
    
    # loop over all angular wedges, estimating fresnel radius for each
    for xi,angle in enumerate(angle_bins):
        l_angle, r_angle = angle, angle+interval
        
        if r_angle > 360: 
            r_angle -= 360
            wedge_idx = np.where(phi >= l_angle) and np.where(phi < r_angle)
            r_sel, d_sel = rho.copy()[wedge_idx], data.copy()[wedge_idx]
            r_fresnel, r_edge = fresnel_radius(r_f, r_sel, d_sel, sigma=sigma)
            mask[np.where((rho<r_fresnel) & (phi>=l_angle))] = 1
            mask[np.where((rho<r_fresnel) & (phi<r_angle))] = 1
        
        else:
            wedge_idx = np.where((phi >= l_angle) & (phi < r_angle))
            r_sel, d_sel = rho.copy()[wedge_idx], data.copy()[wedge_idx]
            r_fresnel, r_edge = fresnel_radius(r_f, r_sel, d_sel, sigma=sigma)
            mask[np.where((rho < r_fresnel) & (phi >= l_angle) & (phi < r_angle))] = 1
                    
        phi_vals[xi], rf_vals[xi], re_vals[xi] = angle + 0.5*interval, r_fresnel, r_edge
        
    return phi_vals, rf_vals, re_vals, mask


def process_image(args, fname, intermediates=False):
    """
    Generate a mask based on fitting the Fresnel fringes for input image.
    
    Inputs:
    -------
    args: dict containing parameters used for Fresnel fitting
    fname: file name of image to proces
    intermediates: if True, return intermediate outputs useful for debugging
    
    Outputs:
    --------
    Note: if intermediates is True, save f_mask to np.array and return None
    f_mask: mask, where 1 corresponds to illuminated, Fresnel-unaffected pixels, optional
    d_interim = dict containing various intermediate outputs for debugging, optional return
    """
    
    # load projection image data
    data = mrcfile.open(fname).data
    
    # fit circle to estimate beam parameters: radius, center 
    mdata = np.zeros_like(data)
    mdata[data>args['b_mask']] = 1
    h_f, k_f, r_f = fit_beam(mdata)
    
    # convert pixel positions to polar coordinates
    rho,phi = cart2pol(data.shape,(h_f,k_f))
    phi += 180
    
    # estimate fresnel radius for full circumference of beam
    masks, phi_s, rf_s, re_s = dict(), np.empty(0), np.empty(0), np.empty(0)
    for num,offset in enumerate(np.arange(0,args['interval'],args['step_size'])):
        phi_o, rf_o, re_o, masks[num] = fit_by_wedge(data, rho, phi, r_f, offset, 
                                                     interval=args['interval'], sigma=args['sigma'])
        phi_s, rf_s, re_s = np.concatenate((phi_s,phi_o)), np.concatenate((rf_s,rf_o)), np.concatenate((re_s,re_o))
        
    # wrap and sort by phi values
    phi_f, rf_f, re_f = phi_s.copy(), rf_s.copy(), re_s.copy()
    phi_f = np.concatenate((phi_f[phi_s>360] - 360, phi_f))
    rf_f = np.concatenate((rf_f[phi_s>360], rf_f))
    re_f = np.concatenate((re_f[phi_s>360], re_f))
    phi_f, rf_f, re_f = phi_f[np.argsort(phi_f)], rf_f[np.argsort(phi_f)], re_f[np.argsort(phi_f)]

    # apply wiener filter to smooth the fresnel and edge radius results
    rf_w = scipy.signal.wiener(rf_f, mysize=args['window'])
    re_w = scipy.signal.wiener(re_f, mysize=args['window'])    
    
    # generate conensus mask by interpolating the wiener-filtered results
    f_mask = np.zeros_like(data) 
    f = scipy.interpolate.interp1d(phi_f, rf_w)

    for angle in np.arange(0,360,args['interval']):
        r_fresnel = f(angle + 0.5*args['interval'])
        f_mask[np.where((rho<r_fresnel) & (phi >= angle) & (phi < angle + args['interval']))] = 1
        
    # optionally return consensus mask and intermediate outputs or just save former
    if intermediates:
        d_interim = dict()
        d_interim['masks'] = masks # all masks
        d_interim['rf_w'], d_interim['re_w'] = rf_w, re_w # wiener filtered arrays
        d_interim['phi_f'], d_interim['rf_f'], d_interim['re_f'] = phi_f, rf_f, re_f
        return f_mask, d_interim
    
    else:
        savename = os.path.join(args['output'], fname.split("/")[-1][:-3] + "npy")
        np.save(savename, f_mask)
        print(f"Saved to {savename}")
        return 


############################################
# Process images in a parallelized fashion #
############################################

def wrap_process_image(args_eval):
    """
    Wrapper function for parallelization with pathos.
    """
    return process_image(*args_eval)


if __name__ == '__main__':

    start_time = time.time()

    # parse command line input and set up output directory as needed
    args = parse_commandline()
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])
    
    # process all images, parallelizing calculation via pathos
    fnames_list = natsorted(glob.glob(args['input']))
    pool = pp.ProcessPool(args['n_processes'])
    args_eval = zip([args]*len(fnames_list), fnames_list)
    pool.map(wrap_process_image, args_eval)

    print(f"elapsed time is {(time.time()-start_time)/60.0:.2f} minutes")

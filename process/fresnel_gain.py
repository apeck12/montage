import scipy.optimize, skimage.filters
from collections import OrderedDict
from natsort import natsorted
import os, glob, time, argparse, sys
import mrcfile, itertools, utils
import numpy as np

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Generate a mask in the shape of each tile for removal of residual Frensel fringes and the
unilluminated region (value of 0) and that optionally corrects radial uneven illumination.
Also save the optimized beam fit parameters: (xc, yc, radius) and diagnostic plots. If a 
single tile cannot be fit, no masks are generated since all tiles are required to stitch a
given tilt angle. This script should be run separately for each tilt angle.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Generate masks to correct Fresnel fringes and uneven illumination.')
    parser.add_argument('-i','--image_paths', help='Path to images collected at a given tilt angle in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-ls','--low_sigma', help='Sigma value for lower kernel of bandpass filter',
                        required=False, type=float, default=2)
    parser.add_argument('-hs','--high_sigma', help='Sigma value for upper kernel of bandpass filter',
                        required=False, type=float, default=3)
    parser.add_argument('-f','--fraction', help='Fraction of max. filtered intensity to estimate fringe ring',
                        required=False, type=float, default=0.2)
    parser.add_argument('-fm','--f_max', help='Fraction of max. filtered intensity for circle fitting, as linspace',
                        required=True, type=float, nargs=3)
    parser.add_argument('-gc','--gain_correct', help='Whether to apply a gain correction',
                        action='store_true') # defaults to False if argument is not supplied
    parser.add_argument('-d','--diagnostics', help='Generate diagnostic plots',
                        action='store_true')
    parser.add_argument('-o','--save_dir', help='Directory to which to save output files',
                        required=True, type=str)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and make new directories as needed.
    """
    # expand paths keys into ordered lists
    args['image_paths'] = natsorted(glob.glob(args['image_paths']))
    
    # create output directory if doesn't already exist
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])

    args['f_max'] = np.linspace(args['f_max'][0], args['f_max'][1], int(args['f_max'][2]))[::-1]

    return args


def cart2pol(shape, center):
    """
    Compute polar coordinates for a 2d grid of specified dimensions.
    
    Parameters
    ----------
    shape : tuple 
        dimensions of 2d grid
    center : tuple
        (x,y) positions of circle's center
    
    Returns
    -------
    rho : numpy.ndarray, shape (M,N)
        radius of each pixel on grid in units of pixels
    phi : numpy.ndarray, shape (M,N) 
        relative polar angle of each pixel, in degrees
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
    
    Parameters
    ----------
    radii : numpy.ndarray of shape (M,) 
        radius values
    intensities : numpy.ndarray of shape (M,) 
        intensities associated with radii array
    
    Returns
    -------
    rx : numpy.ndarray of shape (N,)   
        radius of rprofile array
    rprofile : numpy.ndarray of shape (N,)
        mean radial profile of image
    """
    r = radii.copy().astype(np.int)
    rx = np.unique(r.ravel())
    tbin = np.bincount(r.ravel(), intensities.ravel())
    nr = np.bincount(r.ravel())
    rprofile = tbin[nr!=0] / nr[nr!=0]
    return rx, rprofile 


def compute_fresnel_edge(image_path, low_sigma=2, high_sigma=3, fraction=0.2, f_max=0.5):
    """
    Compute the start of the Fresnel region by bandpass-filtering the tile, followed by 
    fitting a circle to the resulting ring of high-intensity pixels at the edge of the 
    illuminated region and determining the inner radius of this ring.
    
    Parameters
    ----------
    image_path : string 
        path to tile in .mrc format
    low_sigma : float 
        standard deviation of smaller Gaussian kernel for bandpass filtering in pixels
    high_sigma : float
        standard deviation of larger Gaussian kernel for bandpass filtering in pixels
    fraction : float
        fraction of the max filtered intensity for thresholding pixels in ring
    f_max : float
        fraction of max intensity of the filtered tile to keep for circle fitting

    Returns
    -------
    p_opt : tuple 
        optimized beam parameters (h,k,r); if the tile could not be fit, zeros are returned
    """
    def fit_circle(params):
        """
        Cost function for fitting a circle to points; parameters are center and radius.
        """
        xc, yc, r = params
        r_p = np.sqrt((x-xc)**2 + (y-yc)**2)
        return r - r_p

    # load tile, apply bandpass filter, select high-intensity pixels
    tile = mrcfile.open(image_path).data.copy()
    tile = skimage.filters.difference_of_gaussians(tile, low_sigma, high_sigma)
    x,y = np.where(tile>=f_max*tile.max())

    # eliminate pixels too near short edge of detector
    x,y = x[x>10], y[x>10]
    x,y = x[x<tile.shape[0]-10], y[x<tile.shape[0]-10]
    
    # compute initial parameters and fit circle 
    xi,yi,ri = np.median(x), np.median(y), tile.shape[0]/2
    if len(x) > 3:
        p_opt, ier = scipy.optimize.leastsq(fit_circle, (xi,yi,ri))
    else:
        print("Warning: insufficient points, fitting Fresnel region to 0.95*radius")
        p_opt = (xi,yi,0.95*ri) 
    
    # return array of zeros if circle can't be properly fit
    if (p_opt[0] < 0.25*tile.shape[0]) or (p_opt[0] > 0.75*tile.shape[0]):
        return np.zeros(3)

    # remove outliers within 0.9x unoptimized radius to focus on edge region
    rho_i, phi_i = cart2pol(tile.shape, (p_opt[1],p_opt[0]))
    r_mask = np.zeros_like(tile)
    r_mask[rho_i<0.9*ri] = 1
    r_threshold = r_mask[x,y]
    x,y = x[r_threshold==0], y[r_threshold==0]

    # refit circle, assuming sufficient points
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

    if (p_opt[2] < 0.25*tile.shape[0]) or (p_opt[2] > 0.75*tile.shape[0]):
        return np.zeros(3) 
    else:
        return p_opt


def compute_gain_reference(opt_beam, image_paths, kernel_size=None):
    """
    Compute a gain reference, assuming that the gain is characterized by a gradual
    reduction in radial intensity. The reference is computed from the mean radial 
    intensity profiles of all tiles for a particular tilt angle.
    
    Parameters
    ----------
    opt_beam : dictionary 
        optimized beam parameters (h,k,r) for each tile
    image_paths : list of strings
        paths to tiles in mrc format
    kernel_size : float
        size of kernel for median filtering intensity profile, optional
    
    Returns
    -------
    gain_x : numpy.ndarray of shape (N,) 
        radii in pixels associated with gain_y values
    gain_y : numpy.ndarray of shape (N,)
        median-filtered average gain correction profile
    rx : dictionary
        tile index: radii in pixels for radial profiles
    rp : dictionary 
        tile index: intensities for radial profiles
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


def plot_profiles(rx, rp, rx_i, rp_i, rx_c, rp_c, gain_x, gain_y, opt_params, save_dir):
    """
    Plot radial profiles of original and gain-corrected / masked tiles in addition to
    the normalized and gain correction profiles. Save as a .png file.
    """

    f_edge = np.mean(np.array(list(opt_params.values()))[:,2])
    for tag,xstart in zip(['','_inset'],[0.2,0.95]):
    
        f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18, 4))

        for key in rx.keys():
            ax1.plot(rx[key][rx[key]<=opt_params[xi][2]],
                     rp[key][rx[key]<=opt_params[xi][2]], c='black')
            ax2.plot(rx_i[key][rx_i[key]<=opt_params[xi][2]],
                     rp_i[key][rx_i[key]<=opt_params[xi][2]], c='black')
            ax3.plot(rx_c[key], rp_c[key], c='black')

            ax1.plot(rx[key][rx[key]>=opt_params[xi][2]],
                     rp[key][rx[key]>=opt_params[xi][2]], c='grey')
            ax2.plot(rx_i[key][rx_i[key]>=opt_params[xi][2]], 
                     rp_i[key][rx_i[key]>=opt_params[xi][2]], c='grey')
        
        ax1.plot(gain_x, gain_y, c='red', label='Gain correction')
        ax1.set_ylim(0,1.5)
        ax1.set_ylabel("Radial intensity", fontsize=12)

        max_val = np.concatenate([vals for vals in rp_c.values()]).max()
        for ax in [ax1,ax2,ax3]:
            ax.set_xlabel("Radius (pixels)", fontsize=12)
            ax.set_xlim(xstart*f_edge, 1.05*f_edge)
            if ax != ax1:
                ax.set_ylim(0,1.05*max_val)

        ax1.legend(loc=3, fontsize=12)

        ax1.set_title("Normalized profiles and gain", fontsize=14)
        ax2.set_title("Uncorrected profiles", fontsize=14)
        ax3.set_title("Corrected profiles", fontsize=14)

        f.savefig(os.path.join(save_dir, f"masked_profiles{tag}.png"), dpi=300, bbox_inches='tight')
    
    return


def plot_rep_tiles(args, opt_params):
    """
    Plot portions of representative tiles: raw, bandpass-filtered, and masked.
    """  
    indices = np.random.randint(0, len(args['image_paths']), 2)
    args['mask_paths'] = natsorted(glob.glob(os.path.join(args['save_dir'], "*mrc")))

    f = plt.figure(figsize=(14,8))
    for i in range(6):
        xi = indices[i%2]
        hspan = 0.05*opt_params[xi][2]
        f_edge = opt_params[xi][2] + opt_params[xi][1]

        f.add_subplot(3,2,i+1)
        if int(i/2)==0: 
            tile = mrcfile.open(args['image_paths'][xi]).data
            plt.imshow(tile, cmap='Blues', vmax=tile.max())
            plt.plot([f_edge, f_edge],[opt_params[xi][2]-hspan,opt_params[xi][2]+hspan],
                     linestyle='dashed', c='black')
        if int(i/2)==1:
            tile = mrcfile.open(args['image_paths'][xi]).data
            ftile = skimage.filters.difference_of_gaussians(tile, args['low_sigma'], args['high_sigma'])
            plt.imshow(ftile, cmap='Blues', vmax=0.9*ftile.max())
        if int(i/2)==2:
            tile = mrcfile.open(args['image_paths'][xi]).data
            mask = mrcfile.open(args['mask_paths'][xi]).data
            plt.imshow(mask*tile, cmap='Blues', vmax=tile.max())

        plt.xlim(opt_params[xi][1]+0.8*opt_params[xi][2],opt_params[xi][1]+1.1*opt_params[xi][2])
        plt.ylim(opt_params[xi][2]-hspan,opt_params[xi][2]+hspan)

        if i<2: plt.title(f"Representative tile {xi}", fontsize=12)
        if i==0: plt.ylabel("Raw", fontsize=12)
        if i==2: plt.ylabel("Bandpass-filtered", fontsize=12)
        if i==4: plt.ylabel("Masked", fontsize=12)
            
    f.savefig(os.path.join(args['save_dir'], f"rep_tiles.png"), dpi=300, bbox_inches='tight')
    
    return


def plot_beam_fits(args, opt_params):
    """
    Plot beam fits (outer edge of circle) for all tiles.
    """
    f = plt.figure(figsize=(16,16))

    for xi,image_path in enumerate(args['image_paths']):
        if len(args['image_paths']) == 37: # 4 rings
            ax = f.add_subplot(8, 5, xi+1)
        elif len(args['image_paths']) == 61: # 5 rings
            ax = f.add_subplot(9, 7, xi+1)
        else:
            nrows = int(np.around(len(args['image_paths']) / (len(args['image_paths'])%8) + 1))
            ax = f.add_subplot(nrows, (len(args['image_paths'])%8), xi+1)

        tile = mrcfile.open(image_path).data
        plt.imshow(tile)

        thetas = np.deg2rad(np.arange(360))
        c_x, c_y = opt_params[xi][2]*np.cos(thetas), opt_params[xi][2]*np.sin(thetas)
        plt.scatter(c_x+opt_params[xi][1], c_y+opt_params[xi][0], s=0.5, c='white')
        plt.xlim(0,tile.shape[1])
        plt.ylim(0,tile.shape[0])

        plt.text(0.9*tile.shape[1], 0.9*tile.shape[0], f"{xi}", color='white')
        
    f.savefig(os.path.join(args['save_dir'], f"beam_fits.png"), dpi=300, bbox_inches='tight')

    return


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    # fit beam and start of fresnel region
    opt_params = OrderedDict()
    for xi,image_path in enumerate(args['image_paths']):
        print(f"On image {xi}")
        c_params = np.zeros(3)
        for fm in args['f_max']:
            if np.all(c_params==0):
                print(f"Trying f_max {fm}")
                c_params = compute_fresnel_edge(image_path,
                                                low_sigma=args['low_sigma'],
                                                high_sigma=args['high_sigma'],
                                                fraction=args['fraction'],
                                                f_max=fm)
            opt_params[xi] = c_params


    # do not proceed if any tiles cannot be fit
    bad_tiles = [key for key in opt_params.keys() if np.all(opt_params[key]==0)]
    if len(bad_tiles) > 0:
        print("Warning! The beam for the following tile indices could not be fit:")
        print(*bad_tiles)
        print("As a result, no masks will be generated since a single bad tile prevents correct stitching.")
        sys.exit()

    # determine gain correction based on optimized beam parameters
    if args['gain_correct'] is True:
        gain_x, gain_y, rx, rp = compute_gain_reference(opt_params, args['image_paths'])
        f = scipy.interpolate.interp1d(gain_x, gain_y)

    # generate storage dictionaries for radial profiles
    if args['diagnostics'] is True:
        rx_i, rp_i = OrderedDict(), OrderedDict()
        rx_c, rp_c = OrderedDict(), OrderedDict()

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
    
            if args['diagnostics']:
                tile = mrcfile.open(args['image_paths'][xi]).data
                rx_i[xi], rp_i[xi] = radial_profile(rho.flatten(), tile)
                rx_c[xi], rp_c[xi] = radial_profile(rho.flatten(), tile*mask)

        else:
            mask[rho>opt_params[xi][2]] = 0
    
        outpath = os.path.join(args['save_dir'], ipath.split('/')[-1] )
        utils.save_mrc(mask, outpath)

    # save the beam fit parameters, ordered by tile index
    np.save(os.path.join(args['save_dir'], "params.npy"), np.array(list(opt_params.values())))

    # generate diagnostics plots
    if args['diagnostics'] is True:
        plot_beam_fits(args, opt_params)
        plot_rep_tiles(args, opt_params)
        if args['gain_correct'] is True:
            plot_profiles(rx, rp, rx_i, rp_i, rx_c, rp_c, gain_x, gain_y, opt_params, args['save_dir'])
    
    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")


if __name__ == '__main__':
    main()

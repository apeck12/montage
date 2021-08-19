from collections import OrderedDict
from natsort import natsorted
import mrcfile, argparse
import glob, time, os, pickle
import numpy as np

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Compile parameters either for beam fits or from defocus estimation for all tiles at 
each tilt angle into a single dictionary. Save to a pickle file whose keys are tilt
angles and values are arrays of either (h,k,r) parameters for circular beam fits or 
(defocus1, defocus2, d_max, astigmatism, CC, high_res) from defocus estimation for
tile 0.... tile n, where n tiles were collected per tilt angle.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Compile all *params.npy from beam or CTF-fitting.')
    parser.add_argument('-i','--input_prefix', help='Shared prefix for directories containing (ctf_)params.npy', 
                        required=True, type=str)
    parser.add_argument('-c','--centers', help='Path to input beam centers file',
                        required=True, type=str)
    parser.add_argument('-re','--reorder', help='Reorder tilts by angle rather than data collection sequence',
                        action='store_true') # defaults to False if argument is not supplied
    parser.add_argument('-o','--output', help='Path to which to save compiled parameters file',
                        required=True, type=str)
    parser.add_argument('-d','--diagnostics', help='Path for diagnostic plots output if compiling beam fits',
                        required=False, type=str)
    parser.add_argument('-t','--rep_tile', help='Representative tile with pixel size and shape information',
                        required=False, type=str)

    return vars(parser.parse_args())


def plot_params(params, savepath, mrcpath=None):
    """
    Plot beam center and radius for all tiles as a function of tilt angle.

    Parameters 
    ----------
    params : dictionary 
        tilt angle: array of tile parameters, where nth row corresponds to nth tile
    savepath : string
        path for saving diagnostic plot
    mrcpath : string
        path to .mrc file whose header contains voxel dimension
    """
    if mrcpath is not None:
        mrc = mrcfile.open(mrcpath)
        shape, voxel_size = mrc.data.shape, float(mrc.voxel_size.x)
        mrc.close()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,3))

    for i,angle in enumerate(params.keys()):
        h,k,r = params[angle][:,0], params[angle][:,1], params[angle][:,2]
        ax1.scatter(angle*np.ones_like(h), h, c='black', s=2)
        ax2.scatter(angle*np.ones_like(k), k, c='black', s=2)
        ax3.scatter(angle*np.ones_like(r), r, c='black', s=2)

    xs, xf = min(list(params.keys())) - 3, max(list(params.keys())) + 3    
    if mrcpath is not None:
        ax1.plot([xs,xf],[0.5*shape[0],0.5*shape[0]], linestyle='dashed', c='red')    
        ax2.plot([xs,xf],[0.5*shape[1],0.5*shape[1]], linestyle='dashed', c='red')    
        ax3.plot([xs,xf],[0.5*shape[0],0.5*shape[0]], linestyle='dashed', c='red')    
        ax1.set_ylabel(f"Pixels ({voxel_size:.1f} A/px)", fontsize=12)
    else:
        ax1.set_ylabel("Pixels", fontsize=12)

    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel("Tilt angle ($^{\circ}$)", fontsize=12)
        ax.set_xlim(xs,xf)
    
    ax1.set_title("Beam center, X coordinate", fontsize=12)
    ax2.set_title("Beam center, Y coordinate", fontsize=12)
    ax3.set_title("Beam radius", fontsize=12)

    f.savefig(savepath, dpi=300, bbox_inches='tight')

    return


def retrieve_tilts(centers_file):
    """
    Retrieve tilt angles that were collected from input file to SerialEM.
    
    Parameters
    ----------
    centers_file : string 
        filename of predicted beam centers, where x,y coordinates of each 
        tile are listed on separate lines after a particular tilt angle
    
    Returns
    -------
    tilt_angles : numpy.ndarray, shape (N,) 
        tilt angles ordered as images were collected
    """
    
    tilt_angles = list()
    f = open(centers_file, 'r') 
    content = f.readlines() 
    
    # extract tilt angles only
    for line in content:
        as_array = np.array(line.strip().split()).astype(float)
        if (len(as_array) == 1):
            tilt_angles.append(as_array[0])
            
    return np.array(tilt_angles)


def compile_params(args):
    """
    Compile parameters into a single dictionary.

    Parameters
    ----------
    args : dictionary
        command line inputs

    Returns
    -------
    params : dictionary 
        tilt angle: array of tile parameters, where nth row corresponds to nth tile
    """
    
    # retrieve tilt angles
    tilts = retrieve_tilts(args['centers'])

    # optionally reorder tilt series by angle; otherwise by order of collection
    if args['reorder'] is True:
        tilts = sorted(tilts)
    
    # compile all params files
    params = OrderedDict()
    for xi,t in enumerate(tilts):
        try:
            params[t] = np.load(args['input_prefix'] + f"{int(t)}/*params.npy")
        except IOError:
            print(f"Warning: no parameters file found for tilt angle {int(t)}")

    # save to output
    with open(args['output'], "wb") as handle:
        pickle.dump(params, handle)

    return params


def main():

    start_time = time.time()

    args = parse_commandline()
    params = compile_params(args)
    
    if args['diagnostics'] is not None:
        if params[list(params.keys())[0]].shape[1] == 2: # make sure these are beam, not CTF fits
            plot_params(params, args['diagnostics'], mrcpath=args['rep_tile'])

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f}")


if __name__ == '__main__':
    main()

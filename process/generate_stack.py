from collections import OrderedDict
from natsort import natsorted
import mrcfile, argparse, pickle
import glob, time, os, utils
import numpy as np

"""
Compile all stitched tiles together into a tilt-series and save in MRC format. Tiles are 
ordered as projection images were collected unless command line argument reorder is True,
in which case the order is from the most negative to positive tilt angle. Also optionally 
save a tilt angles file (.tlt) for use in downstream reconstruction. 
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Generate a tilt-series from a set of sttiched tiles.')
    parser.add_argument('-i','--stitched_prefix', help='Path to prefix file name of stitched images', 
                        required=True, type=str)
    parser.add_argument('-v','--voxel_path', help='Path to MRC file with voxel size in header',
                        required=False, type=str)
    parser.add_argument('-c','--centers', help='Path to input beam centers file',
                        required=True, type=str)
    parser.add_argument('-p','--params', help='Path to circle-fitting parameters dictionary',
                        required=True, type=str)
    parser.add_argument('-o','--output', help='Output path for tilt stack',
                        required=True, type=str)
    parser.add_argument('-w','--width', help='Length in pixels of retained area of each stitched image',
                        required=True, type=int)
    parser.add_argument('-r','--rotation', help='Global rotation to apply to all beam centers',
                        required=False, type=float, default=0)
    parser.add_argument('-re','--reorder', help='Reorder tilts by increasing angle rather than by data collection',
                        action='store_true') # defaults to False if argument is not supplied
    parser.add_argument('-t','--tilt_file', help='Output path for tilt angles file, ordered as tilt series',
                        required=False, type=str)
    parser.add_argument('-e','--exclude_angles', help='List of tilt angles to exclude (space-separated)',
                        required=False, nargs='+', type=int)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.
    """
    
    if args['voxel_path'] is None:
        args['voxel_path'] = args['stitched_prefix'] + "0.mrc"
    mrc = mrcfile.open(args['voxel_path'])
    args['voxel_size'] = float(mrc.voxel_size.x) # Angstrom / pixel
    mrc.close()

    args['params'] = pickle.load(open(args['params'], "rb"))

    if args['exclude_angles'] is None:
        args['exclude_angles'] = np.array(list())

    return args


def retrieve_beam_centers(centers_file, voxel_size):
    """
    Retrieve the position of the central tile for each tilt angle and convert from 
    microns to pixels
    
    Parameters
    ----------
    centers_file : string 
        filename of predicted beam centers, where x,y coordinates of each 
        tile are listed in um on separate lines after the relevant tilt angle
    voxel_size : float 
        voxel dimensions in A/pixel
    
    Returns
    -------
    origin_shifts : numpy.ndarray, shape (N, 2) 
        global origin shifts applied to each tilt angle
    tilt_angles : numpy.ndarray, shape (N,) 
        tilt angles ordered as images were collected
    """
    
    origin_shifts, tilt_angles = list(), list()
    f = open(centers_file, 'r') 
    content = f.readlines() 
    
    # extract position of tile 0 for each tilt angle
    for line in content:
        as_array = np.array(line.strip().split()).astype(float)
        if (len(as_array) == 1):
            tilt_angles.append(as_array[0])
            counter = 0
        elif (len(as_array) >= 2) and (counter==0):
            origin_shifts.append(as_array * 1e4/voxel_size)
            counter += 1
            
    origin_shifts = np.array(origin_shifts)[:,:2]     
    return origin_shifts, np.array(tilt_angles)


def stack_stitched(args):
    """
    Crop and then stack stitched tilt images into a tilt-series, accounting
    for the global offset in the tile positions between tilt angles.

    Parameters
    ----------
    args: dictionary 
        command line arguments

    Returns
    -------
    tilt_series : numpy.ndarray, shape (n_tilts, N, N) 
        coarsely-aligned tilt-stack
    tilts : numpy.ndarray, shape (N,) 
        tilt angles ordered as images in tilt_series
    """
    # retrieve origin shifts and tilt angles
    shifts, all_tilts = retrieve_beam_centers(args['centers'], args['voxel_size'])
    shifts = utils.apply_rotation(shifts, args['rotation'])
    shifts = np.fliplr(shifts)
    t_shifts = OrderedDict((key,val) for key,val in zip(all_tilts,shifts))
    
    # retrieve all processed tilts and optionally reorder tilt series by angle
    tilts = np.array(list(args['params'].keys()))
    if args['reorder'] is True:
        tilts = np.array(sorted(tilts))
    retained_tilts = np.setdiff1d(tilts, args['exclude_angles'])

    # translational offsets due to spiral
    offsets = np.array([t_shifts[xi] for xi in retained_tilts])
    
    # generate empty tilt-series array
    tilt_series = np.zeros((len(retained_tilts),args['width']*2,args['width']*2))
    
    for xi,t in enumerate(retained_tilts):
        # load image and retrieve center coordinates
        image = mrcfile.open(args['stitched_prefix'] + f"{int(t)}.mrc").data
        xc, yc = (np.array(image.shape)/2).astype(int)

        # spiral translational offsets, accounting for change due to projection
        x_offset, y_offset = np.around(offsets[xi]).astype(int)
        x_offset *= np.cos(np.deg2rad(t))
        x_offset = np.around(x_offset).astype(int)

        tilt_series[xi] = image[xc-args['width']-x_offset: xc+args['width']-x_offset,
                                yc-args['width']-y_offset: yc+args['width']-y_offset]        

    return tilt_series, retained_tilts


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    # generate tilt stack and save to .mrc format
    tseries, retained_tilts = stack_stitched(args)
    utils.save_mrc(tseries, args['output'], args['voxel_size'])

    # optionally save a corresponding .tlt file for IMOD
    if args['tilt_file'] is not None:
        np.savetxt(args['tilt_file'], retained_tilts, fmt="%i", delimiter='\n')

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f} minutes")


if __name__ == '__main__':
    main()

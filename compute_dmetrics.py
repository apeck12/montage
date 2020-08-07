from collections import OrderedDict
from natsort import natsorted
import pandas as pd
import numpy as np
import argparse, os, itertools
import scipy.stats, time, glob

"""
Compute dose metrics from normalized counts files and convert to a pandas DataFrame.
Include the parameters used for each simulation run in the DataFrame.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Compute dose metrics from normalized counts files.')
    parser.add_argument('-i','--input', help='Directory containing normalized counts arrays', 
                        required=True, type=str)
    parser.add_argument('-p','--params', help='Text file with each row indicating a set of simulation parameters',
                        required=True, type=str)
    parser.add_argument('-t','--pattern', help='Pattern type: spiral or snowflake or sunflower', 
                        required=True, type=str)
    parser.add_argument('-s','--savepath', help='Path to which to save pandas DataFrame', 
                        required=True, type=str)
    return vars(parser.parse_args())


def compute_metrics(args):
    """
    Compute dose metrics for each normalized counts file and store in pandas DataFrame, 
    along with the parameters associated with each run. The index matches 'sid'.
    
    Inputs:
    -------
    args: dict containing command line inputs
    
    Outputs:
    --------
    df: pandas DataFrame containing parameters and scoring metrics
    """

    data = OrderedDict()

    # extract parameters: labels and values
    if args['pattern'] == 'spiral':
        labels = ['sid','t_max','x_scale','start_angle','rot_step','alt','con','n_rev','sigma']
        for i,f in enumerate(labels):
            if f in ['x_scale','alt','con']: 
                data[f] = np.genfromtxt(args['params'], usecols=[i], dtype=bool)
            else:
                data[f] = np.loadtxt(args['params'], usecols=[i])
        data.pop('sid')

    elif args['pattern'] == 'snowflake':
        labels = ['sid','t_max','x_scale','start_angle','rot_step','alt','n_steps','sigma']
        for i,f in enumerate(labels):
            if f in ['x_scale','alt']:
                data[f] = np.genfromtxt(args['params'], usecols=[i], dtype=bool)
            else:
                data[f] = np.loadtxt(args['params'], usecols=[i])
        data.pop('sid')

    elif args['pattern'] == 'sunflower':
        labels = ['sid','t_max','x_scale','start_angle','rot_step','alt','con','sigma']
        for i,f in enumerate(labels):
            if f in ['x_scale','alt','con']:
                data[f] = np.genfromtxt(args['params'], usecols=[i], dtype=bool)
            else:
                data[f] = np.loadtxt(args['params'], usecols=[i])
        data.pop('sid')

    else:        
        print("Pattern type not recognized, should be spiral, snowflake, or sunfloewr")
        sys.exit()

    # compute various dose metrics
    fnames = natsorted(glob.glob(os.path.join(args['input'], "*.npy")))
    for key in ['d_max', 'd_var', 'd_mean', 'd_fs_mean', 'd_fa_1.5', 'd_skew', 'n_voxels','f_missing']:
        data[key] = np.zeros(len(fnames))

    for i,f in enumerate(fnames):
        print(f"Computing metrics for norm counts file {i}")
        temp = np.load(f, mmap_mode='r')
        data['d_max'][i] = np.max(temp) # max normalized dose
        data['d_var'][i] = np.var(temp) # variance of distribution
        data['d_mean'][i] = np.mean(temp) # mean normalized dose
        data['d_fs_mean'][i] = len(np.where(temp < np.mean(temp))[0]) / len(temp) # fraction below mean
        data['d_fa_1.5'][i] = len(np.where(temp > 1.5)[0]) / len(temp) # fraction above normalized counts of 1.5
        data['d_skew'][i] = scipy.stats.skew(temp) # skew of distribution
        data['f_missing'][i] = len(np.where(temp < 1.0)[0]) / len(temp) # fraction of missing voxels
        data['n_voxels'][i] = len(temp) # number of voxels

    # convert to pandas DataFrame
    print("Saving to pandas DataFrame")
    df = pd.DataFrame(data)
    
    return df


if __name__ == '__main__':

    start_time = time.time()
    
    args = parse_commandline()
    df = compute_metrics(args)
    df.to_pickle(args['savepath'])

    elapsed_time = (time.time() - start_time)/60.0
    print(f"Elapsed time is {elapsed_time} minutes.")

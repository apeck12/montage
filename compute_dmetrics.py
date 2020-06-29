from collections import OrderedDict
from natsort import natsorted
import pandas as pd
import numpy as np
import argparse, os, itertools
import params, time, glob

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
    parser.add_argument('-p','--pattern', help='Pattern type: spiral or snowflake', 
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
    p = params.set_params(args['pattern'])
    combinations = params.generate_combinations(p, args['pattern'])

    labels = list(p.keys())
    if args['pattern'] == 'spiral': labels.insert(5, "continuous")
    vals = list(combinations.values())
    for i,f in enumerate(labels):
        data[f] = [v[i] for v in vals]

    # compute dose metrics: mean, median, max, variance
    fnames = natsorted(glob.glob(os.path.join(args['input'], "*.npy")))
    for key in ['d_max', 'd_var', 'd_mean', 'd_fs_mean', 'd_fa_1.5', 'n_voxels']:
        data[key] = np.zeros(len(fnames))

    for i,f in enumerate(fnames):
        print(f"Computing metrics for norm counts file {i}")
        temp = np.load(f)
        data['d_max'][i], data['d_var'][i] = np.max(temp), np.var(temp)
        data['d_mean'][i] = np.mean(temp)
        data['d_fs_mean'][i] = len(np.where(temp < np.mean(temp))[0]) / len(temp) # fraction below mean
        data['d_fa_1.5'][i] = len(np.where(temp > 1.5)[0]) / len(temp) # fraction above normalized counts of 1.5
        data['n_voxels'][i] = len(temp)

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

from natsort import natsorted
import os, glob, time, argparse
import mrcfile, utils
import numpy as np

"""
Wrapper for binning a series of projection images using IMOD's newstack function.
Input images can be in either mrc or npy format but will be saved as mrc files.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Bin a set of projection images.')
    parser.add_argument('-i','--image_paths', help='Path to images in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-b','--bin_factor', help='Factor by which to bin images',
                        required=True, type=int)
    parser.add_argument('-o','--output', help='Output directory for saving binned images',
                        required=True, type=str)

    return vars(parser.parse_args())


def modify_args(args):
    """
    Modify command line arguments and add additional information to keys.

    Parameters
    -----------
    args : dictionary 
        contains image_paths and output keys

    Returns
    --------
    args : dictionary
        input modified, with filenames glob-expanded and data type noted
    """
    # get data type and expand paths keys into ordered lists
    args['dtype'] = args['image_paths'][-3:] 
    args['image_paths'] = natsorted(glob.glob(args['image_paths']))
    
    # create output directory if doesn't already exist
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])
    
    return args


def bin_all_images(args):
    """
    Bin all images by desired factor using the newstack function in IMOD.
    
    Parameters
    -----------
    args : dictionary
        contains image paths, bin factor, data type, and output path
    """

    bin_factor = args['bin_factor']

    if args['dtype'] == 'mrc':
        for i,fname in enumerate(args['image_paths']):
            outpath = os.path.join(args['output'], fname.split('/')[-1])
            os.system(f'newstack -input {fname} -output {outpath} -bin {bin_factor}')

    elif args['dtype'] == 'npy':
        temp_path = os.path.join(args['output'], "temp.mrc")

        for i,fname in enumerate(args['image_paths']):
            # load and convert to temporarily-stored MRC file
            m = np.load(fname)
            utils.save_mrc(m, temp_path)

            # bin image using newstack
            nname = fname.split('/')[-1][:-3] + 'mrc' 
            outpath = os.path.join(args['output'], nname)
            os.system(f'newstack -input {temp_path} -output {outpath} -bin {bin_factor}')
            os.system(f'rm {temp_path}')
    
    else:
        print("Unrecognized file type; should be .mrc or .npy")
        
    return


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)
    bin_all_images(args)

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f} minutes")


if __name__ == '__main__':
    main()


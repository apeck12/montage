from natsort import natsorted
import os, glob, time, argparse
import mrcfile, pickle, subprocess
import numpy as np
import utils

"""
Estimate the defocus of the inscribed region of the beam for each tile using CTFFIND4 
and correct using ctfphaseflip from IMOD. Note that the bash scripts this file calls
assumes that ctffind4 version 4.1.13 is available as a module and that IMOD functions
can be called from the command line.
"""

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Use CTFFIND4 to estimate defocus and correct with ctfphasflip.')
    parser.add_argument('-i','--image_paths', help='Path to images in glob-readable format', 
                        required=True, type=str)
    parser.add_argument('-p','--params', help='Path to parameters describing (h,k,r) of each circular tile',
                        required=True, type=str)
    parser.add_argument('-d','--defocus', help='Defocus search parameters: min,max,step in Angstrom',
                        required=True, nargs=3, type=float)
    parser.add_argument('-o','--save_dir', help='Directory to which to save corrected MRC files',
                        required=True, type=str)
    parser.add_argument('-td','--temp_dir', help='Directory to which to save intermediate fitting files',
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
    if not os.path.isdir(args['temp_dir']):
        os.mkdir(args['temp_dir'])
        
    # retrieve parameters
    args['params'] = np.load(args['params'])
        
    # retrieve voxel size from MRC header in Angstrom per pixel
    mrc = mrcfile.open(args['image_paths'][0])
    args['voxel_size'] = float(mrc.voxel_size.x)
    mrc.close()

    return args


def inscribed_region(image_path, c_params):
    """
    Extract the subtile inscribed by the circular illuminated region.
    
    Parameters
    ----------
    image_path : string
        path to recorded tile
    c_params : numpy.ndarray, shape (3,)
        (h,k,r) - center and radius of illuminated region
        
    Returns
    -------
    p_tile : numpy.ndarray, shape (M,N)
        region of tile inscribed in illuminated area
    """
    tile = mrcfile.open(image_path).data.copy()
    xc, yc, r = c_params
    
    x0, x1 = int(xc-r/np.sqrt(2)), int(xc+r/np.sqrt(2))
    y0, y1 = int(yc-r/np.sqrt(2)), int(yc+r/np.sqrt(2))
    
    return tile[x0:x1,y0:y1]


def run_ctffind4(args, filename, outname, high_res):
    """
    Estimate the defocus of input MRC file using CTFFIND4.
    
    Parameters
    ----------
    args : dictionary
        command-line inputs
    filename : string 
        mrc file of inscribed illuminated region
    outname : string
        output MRC filename for CTFFIND4 diagnostic
    high_res : float
        high resolution limit during defocus estimation
    
    Returns
    -------
    def1 : float 
        estimated defocus 1
    def2 : float 
        estimated defocus 2 (two values due to astigmatism)
    d_max : float
        high resolution of good fit to Thon rings according to CTFFIND4
    astig : float
        estimated azimuth of astigmatism in degrees
    cc : float
        mean cross-correlation up to d_max
    """
    
    # run CTFFIND4 
    command=f"./run_ctffind.sh {filename} {outname} {args['voxel_size']} {args['defocus'][0]} {args['defocus'][1]} {args['defocus'][2]} {high_res}"
    subprocess.run(args=[command], shell=True)
    
    # extract fit parameters
    fit = np.loadtxt(outname[:-4] + ".txt")
    def1, def2, astig = fit[1:4]
    d_max = fit[-1]
    fit_avrot = np.loadtxt(outname[:-4] + "_avrot.txt")
    cc_sel = np.mean(fit_avrot[4][np.where(fit_avrot[0]<1.0/20.0)[0]])

    return def1, def2, d_max, astig, cc_sel


def estimate_defocus_batch(args):
    """
    Loop through and process all tiles by estimating defocus from the cropped region.

    Parameters
    ----------
    args : dictionary 
        command line inputs

    Returns
    -------
    ctf_params: numpy.ndarray, shape (n_tiles, 6) 
       rows index tiles, while columns are [defocus1, defocus2, CTFFIND4 d_max, astigmatism, CC fit]
    """

    ctf_params = np.zeros((len(args['image_paths']),6))

    for i in range(len(args['image_paths'])):
    
        # crop input MRC file to illuminated region
        partial = inscribed_region(args['image_paths'][i], args['params'][i])
        filename = os.path.join(args['temp_dir'], f"inscribed.mrc")
        utils.save_mrc(partial, filename)

        # estimate defocus, trying different values of max resolution
        outname = os.path.join(args['temp_dir'], f"fit{i}.mrc")
        high_res = np.array([10,15,20,25,30])
        search_params = np.zeros((len(high_res),5))

        for xi,hr in enumerate(high_res):
            search_params[xi] = run_ctffind4(args, filename, outname, hr)
        ctf_params[i,5] = high_res[np.argmax(search_params[:,-1])]
        ctf_params[i,:5] = run_ctffind4(args, filename, outname, ctf_params[i,5])

        subprocess.run(args=[f"rm {filename}"], shell=True)
        
    return ctf_params


def ctf_correct_batch(args, ctf_params):
    """
    Correct for CTF using ctfphaseflip in all tiles, saving the corrected MRC.

    Parameters
    ----------
    args : dictionary 
        command line inputs
    ctf_params: numpy.ndarray, shape (n_tiles, 6) 
       rows index tiles, while columns are [defocus1, defocus2, CTFFIND4 d_max, astigmatism, CC fit]
    """

    # paths to files used by ctfphaseflip
    defocus_file = os.path.join(args['temp_dir'], "defocus.txt")
    tilt_angle_file = os.path.join(args['temp_dir'], "tilt_angle.txt")
    np.savetxt(tilt_angle_file, np.zeros(1), fmt="%i")

    for i in range(len(args['image_paths'])):

        # create temporary defocus file, converting mean defocus estimate to nm
        est_defocus = ctf_params[i][:2]
        est_defocus_diff = np.abs(np.diff(est_defocus))
        if est_defocus_diff > args['defocus'][2]:
            print(f"Warning, tile {i}: defocus estimates differ by {est_defocus_diff[0]/10/1000:.2f} microns")
        cfp_input = np.array([0, 1, 0, 0, 0]) # see ctfphaseflip guide for details
        cfp_input[-1] = np.mean(est_defocus/10., keepdims=True)
        np.savetxt(defocus_file, cfp_input, fmt='%i', newline=" ")

        # preprocess tile and generate paths
        outname = os.path.join(args['save_dir'], args['image_paths'][i].split('/')[-1])
        filename = args['image_paths'][i]

        # correct for CTF using ctpfphaseflip
        command=f"./run_ctfphaseflip.sh {filename} {outname} {tilt_angle_file} {defocus_file}"
        subprocess.run(args=[command], shell=True)
        subprocess.run(args=[f"rm {defocus_file}"], shell=True)

    return


def process_tiles(args, cc_threshold=0.8):
    """
    Process all tiles in tilt: estimating defocus and refining this estimate using 
    CTFFIND4, followed by CTF correction using ctfphaseflip. Two rounds of defocus
    estimation are performed; during the second round, the defocus values from well
    fit tiles during the first round are used to determine the new defocus range to
    search over.

    Parameters
    ----------
    args : dictionary 
        command line inputs
    cc_threshold : float
        cross-correlation threshold
    """
    
    # first round of defocus estimation
    ctf_params = estimate_defocus_batch(args)
    np.save(os.path.join(args['temp_dir'],"ctf_params_initial.npy"), ctf_params)

    # refine defocus estimation by searching over smaller interval
    reliable_idx = np.where(ctf_params[:,-2] > cc_threshold)[0]
    if len(reliable_idx) == 0:
        print("Warning: CC threshold too high, cannot refine defocus estimation.")
    else:
        df_std = np.std(ctf_params[reliable_idx,:2])
        new_min, new_max = ctf_params[reliable_idx,:2].min() - 3*df_std, ctf_params[reliable_idx,:2].max() + 3*df_std
        args['defocus'] = [new_min, new_max, args['defocus'][2]/2]    
        print(f"New defocus search min, max, step: {args['defocus']}")
        ctf_params = estimate_defocus_batch(args)
        np.save(os.path.join(args['temp_dir'],"ctf_params.npy"), ctf_params)

    # correct CTF
    ctf_correct_batch(args, ctf_params)

    return 


def main():

    start_time = time.time()

    args = parse_commandline()
    args = modify_args(args)

    process_tiles(args)

    print(f"elapsed time is {((time.time()-start_time)/60.0):.2f} minutes")


if __name__ == '__main__':
    main()

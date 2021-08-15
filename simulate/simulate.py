from Microscope_parameters import *
from Microscope_setup import *
import argparse, os, logging, sys
import numpy as np

"""
Simulate specimen exposure during data collection of a montage tilt-series. 
The specimen is modeled as a rectangular prism and discretized to determine
the spatial distribution of dose. The circular beam is tiled on a hexagonal
grid at each tilt angle, and the positions of the beam centers can be offset
between tilt angles by introducing translational and / or rotational offsets.

For the translations, the global offsets can follow one of three basic spiral
patterns: snowflake, classical (Archimedean), or sunflower (Fibonacci). Other
parameters include X-axis scaling and the maximum permitted translation across
the tilt-series. For the rotational offsets, parameters include the rotation
step size, continuous versus alternating, and the initial offset in the plane
of the detector.

The output is a numpy array of dose per specimen voxel, normalized such that 
a value of 1 indicates the amount of dose the voxel would have received in a 
normal tilt-series, assuming it was illuminated throughout data collection.
"""

def str2bool(v):
    """
    Convert string variations of True/False to relevant boolean. Code is courtesy:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_commandline():
    """
    Parse command line input and set default values for inputs not given.
    """
    parser = argparse.ArgumentParser(description='Simulate tilt-series exposure for a given tiling pattern.')
    parser.add_argument('-sid', '--sim_ind', help='Simulation trial id',
                        required=True, type=int)
    parser.add_argument('-n', '--n_processor', help='Number of processors for parallelizing beams',
                        required=True, type=int)
    # parameters defining beam, specimen, and tilt-scheme
    parser.add_argument('-v','--volume_3d', 
                        help='Volume of the specimen: x y z (length in units of nm)', 
                        required=False, default=vol, nargs=3, type=float)
    parser.add_argument('-vs','--voxel_size', 
                        help='Voxel size of specimen, in units of nm', 
                        required=False, default=vol_size, type=float)
    parser.add_argument('-ia','--interested_area', 
                        help='Interested area of the specimen', 
                        required=False, default='all_circles', type=str)
    parser.add_argument('-tr','--tilt_range', help='Range of tilt angles: max min', 
                        required=False, default=tilt_angle_range, nargs=2, type=float)
    parser.add_argument('-ti','--tilt_increment', help='Increment of tilt angle', 
                        required=False, default=tilt_angle_increment, type=float)
    parser.add_argument('-cos','--cosine', help='Whether dose increases by 1/cos(tilt_angle)',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-r','--radius', default=rad, help='Radius of the beam', 
                        required=False, type=float)
    # parameters defining translational offset scheme
    parser.add_argument('-st','--spiral_type',
                        help='Type or spiral pattern: classic, snowflake, or sunflower',
                        required=True, default='classic', type=str)
    parser.add_argument('-tm','--max_trans', 
                        help='Maximum translation length in unit of radius', 
                        required=False, default=2, type=float)
    parser.add_argument('-nr','--n_revolutions',
                        help='Number of total revolutions for spiral pattern',
                        required=False, default=2.0, type=float)
    parser.add_argument('-ns','--n_steps',
                        help='Number of positions from the snowflake center along each radial line',
                        required=False, default=2, type=int)
    parser.add_argument('-x','--xscale', help='X-axis scaling of translational offset',
                        required=False, default=False, type=str2bool)
    # parameters defining rotational offset scheme
    parser.add_argument('-sa','--start_beam_angle', help='Starting angle of the beams', 
                        required=False, default=0, type=float)
    parser.add_argument('-rs','--rotation_step_size', 
                        help='Rotation step size between 0 and 60 degree', 
                        required=False, default=30, type=float)
    parser.add_argument('-al','--alternate', 
                        help='Alternate between +/- rotation from one image to next', 
                        required=False, default=False, type=str2bool)
    parser.add_argument('-co','--continuous',
                        help='Continually increment by rotation_step_size from one image to next',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-nr','--n_revolutions',
                        help='Number of total revolutions for spiral pattern',
                        required=False, default=2.0, type=float)
    # miscellaneous paramters to account for experimental non-idealities
    parser.add_argument('-bid', '--act_beam_ind', help='Indices of non-buffer beams',
                        required=False, default=act_beam_ind, nargs='+', type=int)
    parser.add_argument('-sig', '--shift_sigma', help='Sigma of normal distribution for beam shift errors',
                        required=False, default=0.0, type=float)
    parser.add_argument('-of','--overlap_fraction', 
                        help='Fractional targeted overlap between two adjacent tiles, float between 0 and 1', 
                        required=False, default=cal_fractional_overlap(), type=float)
    parser.add_argument('-ff','--fringe_fraction', 
                        help='Fraction of beam radius discarded due to Fresnel fringes, float between 0 and 1', 
                        required=False, default=0, type=float)
    parser.add_argument('-out', '--out_dir', help='Path for saving output files',
                        required=False, default='./scratch', type=str)
    return vars(parser.parse_args())


def prepare_logger(args):
    """
    Set up logger for output information.

    Parameters:
    -------
    args: dict of command line inputs

    Ouputs:
    -------
    main_logger: logger object
    """
    log_dir = os.path.join(args['out_dir'], 'log_files')
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_name = f"log{args['sim_ind']}_n{args['n_processor']}_{args['voxel_size']}.log"
    logging.basicConfig(filename=os.path.join(log_dir, log_name), filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    main_logger = logging.getLogger('simulator')
    main_logger.setLevel(logging.INFO)
    return main_logger


def generate_voxels_dict(sample, sample_holder, dir_name):
    """
    Precalculate voxel coordinates for tilt-series, saving a np.array per tilt angle,
    unless precomputed files already exist.
    
    Parameters:
    -----------
    sample: instance of Sample class from Microscope_setup.py
    sample_holder: instance of SampleHolder class from Microscope_setup.py
    dir_name: path to which to save precomputed voxels arrays
    """
    if not os.path.exists(dir_name): 
        os.makedirs(dir_name)
        all_angles = sample_holder.all_tilt_angles
        for a,tilt_angle in enumerate(all_angles):
            np_file_name = os.path.join(dir_name, f'vs{sample.voxel_size}nm_{tilt_angle}')
            np.save(np_file_name, sample.voxel_centers[:,:-1])
            sample_holder.raptor_tilt(sample)
        main_logger.info(f'Finished precalculating voxels for each tilt angle at {dir_name}')
    else:
        main_logger.info(f'Voxels for all angles are already calculated in {dir_name}')
            
    return


def prepare_sample(args):
    """
    Prepare instances of Sample and SampleHolder classes, including precalculation
    of voxel coordinates arrays.
    
    Parameters:
    -----------
    args: dictionary of arguments specifying simulation parameters
    
    Returns:
    --------
    sample: instance of Sample class from Microscope_setup
    sample_holder: instance of SampleHolder class from Microscope_setup
    sample_voxel_dir: path to pre-computed voxel coordinates arrays
    """
    # set up instances of Sample and SampleHolder classes
    sample = Sample(volume_3d=args['volume_3d'], voxel_size=args['voxel_size'],
                    interested_area=args['interested_area'])
    sample_holder = SampleHolder(sample, tilt_range=args['tilt_range'],
                                 tilt_increment=args['tilt_increment'])

    # precompute voxel coordinates arrays as needed
    x,y,z = int(sample.x_len), int(sample.y_len), int(sample.z_len)
    sample_voxel_dir = os.path.join(args['out_dir'], f'vol{x}_{y}_{z}_vs{sample.voxel_size}')
    generate_voxels_dict(sample, sample_holder, sample_voxel_dir)

    # called twice to reset sample.voxel_centers to 0 degrees for downstream interested_area??
    sample = Sample(volume_3d=args['volume_3d'], voxel_size=args['voxel_size'], 
                    interested_area=args['interested_area']) 
    
    return sample, sample_holder, sample_voxel_dir


def prepare_beam(args, tilt_angles, cos=False, sigma=0):
    """
    Prepare instance of Beam class, including tiling pattern for each angle of the
    tilt-series through the relevant Beam_offset_generator class. If target ovlerap
    and/or fringe fraction parameters are supplied, ideal beam positions are adjusted
    to accommodate additional overlap. If sigma is non-zero, draw beam shift errors 
    from a normal distribution with sigma standard deviation independently for each 
    beam position's x and y coordinates.
    
    Parameters:
    -----------
    args: dictionary of arguments specifying simulation parameters
    tilt_angles: np.array of ordered angles in tilt-series
    cos: boolean dictating whether dose follows a 1/cos(tilt_angle) scheme
    sigma: standard deviation of normal distribution for beam shift errors

    Returns:
    --------
    beam: instance of Beam class
    """
    # prepare hexagonally-tiled beams, adjusting for Fresnel fringes or greater overlap as needed
    rand_beam_pos, act_beam_ind = hexagonal_tiling(max_one_row=max_one_row, n_interest=n_interest)
    if (args['overlap_fraction'] == cal_fractional_overlap()) and (args['fringe_fraction'] == 0):
        rand_beam_pos *= args['radius']
    else:
        stride = interpolate_stride(args['overlap_fraction'], interp_points=int(1e7), 
                                    radius=(1.0-args['fringe_fraction']), n_overlaps=1)
        rand_beam_pos *= args['radius']*(stride/np.sqrt(3))

    # set up instance of Beam class
    beam = Beam(radius=args['radius'], beam_pos=rand_beam_pos,
                actual_beam_ind=args['act_beam_ind'], n_processor=args['n_processor'], cos=cos)

    # set up instance of Beam_offset_generator class for spiral class of interest
    if args['spiral_type'] == 'classic':
        beam_offset_generator = Beam_offset_generator_spiral(radius=args['radius'], 
                                                             beam_positions=rand_beam_pos,
                                                             tilt_series=tilt_angles,
                                                             n_revolutions=args['n_revolutions'],
                                                             max_trans=args['max_trans'],
                                                             xscale=args['xscale'], 
                                                             start_beam_angle=args['start_beam_angle'], 
                                                             rotation_step=args['rotation_step_size'],
                                                             alternate=args['alternate'],
                                                             continuous=args['continuous'])
    elif args['spiral_type'] == 'sunflower':
        beam_offset_generator = Beam_offset_generator_sunflower(radius=args['radius'], 
                                                                beam_positions=rand_beam_pos,
                                                                tilt_series=tilt_angles,
                                                                max_trans=args['max_trans'],
                                                                xscale=args['xscale'],
                                                                start_beam_angle=args['start_beam_angle'],
                                                                rotation_step=args['rotation_step_size'],
                                                                alternate=args['alternate'],
                                                                continuous=args['continuous'])
    elif args['spiral_type'] == 'snowflake':
        beam_offset_generator = Beam_offset_generator_snowflake(radius=args['radius'], 
                                                                beam_positions=rand_beam_pos,
                                                                tilt_series=tilt_angles,
                                                                max_trans=args['max_trans'],
                                                                xscale=args['xscale'],
                                                                start_beam_angle=args['start_beam_angle'],
                                                                rotation_step=args['rotation_step_size'],
                                                                alternate=args['alternate'],
                                                                n_steps=args['n_steps'])
    else:
        print("Spiral type is not recognized; must be classic, sunflower, or snowflake")
        sys.exit()

    # compute tiling patterns for each image in the tilt-series and store in beam.patterns
    beam_offset_generator.offset_all_beams()
    beam.patterns = beam_offset_generator.offset_patterns

    # add beam shift errors, drawn from a normal distribution
    if sigma != 0:
        for a in beam.patterns.keys():
            beam.patterns[a] += np.random.normal(loc=0.0,scale=sigma,size=beam.patterns[a].shape)
    
    return beam


def simulate_exposure(sample, beam, tilt_angles, sample_voxel_dir, roi_mask=None):
    """
    Simulate exposure for entire tilt-series and compute the normalized exposure
    counts for each specimen voxel.
    
    Parameters:
    -------
    sample: instance of Sample class from Microscope_setup
    beam: instance of Beam class, with tiling patterns pre-computed
    sample_voxel_dir: path to pre-computed voxel coordinates arrays
    tilt_angles: np.array of ordered angles in tilt-series
    roi_mask: boolean array indicating voxels in region of interest, optional.
        If None, set to sample.interest_mask.
    
    Returns:
    --------
    norm_voxel_counts: normalized counts of length no. voxels in act_beam_ind 
    """
    sample.set_interested_area(beam, sample_voxel_dir)
    sample.exposure_counter += beam.image_all_tilts(sample_voxel_dir, 
                                                    tilt_angles, 
                                                    sample.voxel_size) 
    if roi_mask is None: 
        roi_mask = sample.interest_mask
    
    if beam.cos is False:
        return sample.exposure_counter[roi_mask]/len(tilt_angles)
    else:
        max_exposure = np.sum(1.0/np.cos(np.deg2rad(tilt_angles)))
        return sample.exposure_counter[roi_mask]/max_exposure


def main():

    start_time = time.time()

    # retrieve simulation parameters and prepare output directory
    args = parse_commandline()
    if not os.path.exists(args['out_dir']): os.makedirs(args['out_dir'])

    # set up logger
    main_logger = prepare_logger(args)

    # retrieve region of interest from ideal pattern if simulating beam shift errors 
    if args['shift_sigma'] != 0:
        sample_ideal, sample_holder_ideal, sample_voxel_dir = prepare_sample(args)
        beam_ideal = prepare_beam(args, sample_holder_ideal.all_tilt_angles)
        sample_ideal.set_interested_area(beam_ideal, sample_voxel_dir)
        roi = sample_ideal.interest_mask.copy()
    else:
        roi = None

    # simulate exposure for the given tiling pattern
    sample, sample_holder, sample_voxel_dir = prepare_sample(args)
    beam = prepare_beam(args, sample_holder.all_tilt_angles, cos=args['cosine'], sigma=args['shift_sigma'])
    norm_counts = simulate_exposure(sample, beam, sample_holder.all_tilt_angles, sample_voxel_dir, roi_mask=roi)

    # save normalized exposure counts
    save_dir = os.path.join(args['out_dir'], 'norm_counts')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    norm_counts_file = os.path.join(save_dir, f"sid{args['sim_ind']}_vs{sample.voxel_size}")
    np.save(norm_counts_file, norm_counts)

    elapsed_time = (time.time() - start_time)/60.0
    main_logger.info(f'Simulation took {elapsed_time} minutes in total.')


if __name__ == '__main__':
    main()

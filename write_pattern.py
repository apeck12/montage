from Microscope_parameters import *
from Microscope_setup import *
import argparse, os
import numpy as np

"""
Format the tile positions (beam centers) for the specified spiral pattern to a text file,
where each line is space-delimited and contains the following information:
tilt_angle x1 y1 x2 y2 ... xn yn
where tilt_angle is in degrees and the beam centers are in microns. The origin coincides
with the center of the region of interest. For each tilt angle, the beam positions are
ordered such that they spiral outwards from the specimen center.
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
    parser = argparse.ArgumentParser(description='Write specified spiral pattern to a text file.')
    parser.add_argument('-tr','--tilt_range', help='Range of tilt angles: max min', 
                        required=False, default=tilt_angle_range, nargs=2, type=float)
    parser.add_argument('-ti','--tilt_increment', help='Increment of tilt angle', 
                        required=False, default=tilt_angle_increment, type=float)
    parser.add_argument('-mr','--max_one_row', help='Maximum number of tiles in a row',
                        required=False, default=max_one_row, type=int)
    parser.add_argument('-r','--radius', default=rad, help='Radius of the beam', 
                        required=False, type=float)
    parser.add_argument('-tm','--max_trans', 
                        help='Maximum translation length in unit of radius', 
                        required=False, default=2, type=float)
    parser.add_argument('-x','--xscale', help='X-axis scaling of translational offset',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-sa','--start_beam_angle', help='Starting angle of the beams', 
                        required=False, default=0, type=float)
    parser.add_argument('-rs','--rotation_step_size', 
                        help='Rotation step size between 0 and 60 degree', 
                        required=False, default=30, type=float)
    parser.add_argument('-al','--alternate', 
                        help='Alternate between +/- rotation from one image to next', 
                        required=False, default=True, type=str2bool)
    parser.add_argument('-co','--continuous',
                        help='Continually increment by rotation_step_size from one image to next',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-nr','--n_revolutions',
                        help='Number of total revolutions for spiral pattern',
                        required=False, default=2.0, type=float)
    parser.add_argument('-of','--overlap_fraction', 
                        help='Fractional targeted overlap between two adjacent tiles, float between 0 and 1', 
                        required=False, default=cal_fractional_overlap(), type=float)
    parser.add_argument('-ff','--fringe_fraction', 
                        help='Fraction of beam radius discarded due to Fresnel fringes, float between 0 and 1', 
                        required=False, default=0, type=float)
    parser.add_argument('-out', '--outfile', help='File name of output file',
                        required=False, type=str)
    return vars(parser.parse_args())


def hexagonal_base(args):
    """
    Compute the (x,y) coordinates of the beam centers for hexagonal tiling, ordered such that
    they spiral outwards from the origin to minimize shift required between consecutive tiles.

    Inputs:
    -------
    args: dict of command-line inputs
    
    Outputs:
    --------
    ord_beam_pos: 2d array of beam center positions in units of nm.
    """
    from functools import reduce
    import operator, math

    # compute ordered tile positions according to ideal hexagonal tiling 
    r_seq = [(0.0,0.0)] # the first tile is at the origin of the region of interest
    rlimit = 0.1

    for nr in range(3,args['max_one_row']+2,2):
        # compute all positions given 'max_one_row' argument
        rbp, abi = hexagonal_tiling(max_one_row=nr, n_interest=nr)
    
        # eliminate any tile centers that belong to previous layer
        radii = np.sqrt(np.sum(np.square(rbp), axis=1))
        rbp = rbp[radii>1.1*rlimit]    
    
        # reorder the remaining coordinates (i.e. ones in this layer) in clockwise fashion
        # code courtesy: https://stackoverflow.com/questions/51074984/sorting-according-to-clockwise-point-coordinates
        coords = [tuple(rbp[i]) for i in range(rbp.shape[0])]
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add,x,y),coords), [len(coords)]*2))
        r_seq += (sorted(coords,
                         key=lambda coord: (-135-math.degrees(math.atan2(*tuple(map(operator.sub,
                                                                                    coord,
                                                                                    center))[::-1]))) % 360))
        # reset the lower bound of radius for next layer
        rlimit = np.max(radii)

    ord_beam_pos = np.array(r_seq)
    
    # compensate for fresnel-fringe region
    if (args['overlap_fraction'] == cal_fractional_overlap()) and (args['fringe_fraction'] == 0):
        ord_beam_pos *= args['radius']
    else:
        stride = interpolate_stride(args['overlap_fraction'], interp_points=int(1e7), 
                                    radius=(1.0-args['fringe_fraction']), n_overlaps=1)
        ord_beam_pos *= args['radius']*(stride/np.sqrt(3))

    return ord_beam_pos


def generate_spiral(args, ord_beam_pos):
    """
    Generate spiral pattern specified by paramters in command-line dict.

    Inputs:
    -------
    args: command-line input dict
    ord_beam_pos: 2d array of hexagonal tile positions

    Outputs:
    --------
    ts_beam_pos: OrderedDict with keys as tilt angles and values as beam positions
    """
    # get array of ordered tilt angles using Sample and SampleHolder classes
    sample = Sample(volume_3d=vol, voxel_size=vol_size, interested_area="3420_3420")
    sample_holder = SampleHolder(sample, tilt_range=args['tilt_range'], 
                                 tilt_increment=args['tilt_increment'])
    tilt_angles = sample_holder.all_tilt_angles

    # set up instance of Beam_offset_generator_spiral class and offset tile positions
    beam_offset_generator = Beam_offset_generator_spiral(radius=args['radius'],
                                                         beam_positions=ord_beam_pos,
                                                         tilt_series=tilt_angles,
                                                         n_revolutions=args['n_revolutions'],
                                                         max_trans=args['max_trans'],
                                                         xscale=args['xscale'],
                                                         start_beam_angle=args['start_beam_angle'],
                                                         rotation_step=args['rotation_step_size'],
                                                         alternate=args['alternate'],
                                                         continuous=args['continuous'])

    beam_offset_generator.offset_all_beams()
    return beam_offset_generator.offset_patterns


def format_output(ts_beam_pos, savepath):
    """
    Write angles and associated beam positions (in microns) to savepath. 

    Inputs:
    -------
    ts_beam_pos: OrderedDict with keys as tilt angles and values as beam positions
    savepath: file to which to write ts_beam_pos
    """
    f = open(savepath, "w")

    for angle in ts_beam_pos.keys():
        tpos = ts_beam_pos[angle].copy()
        xy_list = [round(coord/1000.0,6) for coord in tpos.flatten()] # convert from nm to microns
        xy_list.insert(0,angle) # insert tilt angle as first item per line
        f.write(' '.join(str(val) for val in xy_list)) # space-delimited
        f.write("\n") # new line between tilt angles
    
    f.close()
    return

if __name__ == '__main__':

    start_time = time.time()

    # retrieve command-line inputs
    args = parse_commandline()
    
    # generate spiral pattern from hexagonal base tiling
    ord_beam_pos = hexagonal_base(args)
    ts_beam_pos = generate_spiral(args, ord_beam_pos)

    # write to text file
    format_output(ts_beam_pos, args['outfile'])

    elapsed_time = (time.time() - start_time)/60.0

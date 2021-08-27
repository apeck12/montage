import numpy as np
import argparse
from sim_parameters import *
from sim_setup import *
from simulate import Beam_offset_generator_spiral, str2bool

"""
For a given (Archimedean/classic) spiral strategy, write the coordinates of all tiles
at each tilt angle to a text file as input for SerialEM. The tile coordinates include
both the x and y positions in the plane of the detector and the z-height at the tile's
center (to enable autofocusing). These coordinates are listed after the relevant angle,
and tiles are ordered to spiral outwards from the central tile. 
"""

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
    parser.add_argument('-tm','--max_trans', help='Maximum translation length in unit of radius', 
                        required=False, default=2, type=float)
    parser.add_argument('-x','--xscale', help='X-axis scaling of translational offset',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-sa','--start_beam_angle', help='Initial angular offset of beams in detector plane', 
                        required=False, default=0, type=float)
    parser.add_argument('-rs','--rotation_step_size', help='Rotation step size between 0 and 60 degrees', 
                        required=False, default=30, type=float)
    parser.add_argument('-al','--alternate', help='Alternate between +/- rotation from one image to next', 
                        required=False, default=True, type=str2bool)
    parser.add_argument('-co','--continuous', help='Continually increment by rotation_step_size from one image to next',
                        required=False, default=False, type=str2bool)
    parser.add_argument('-nr','--n_revolutions', help='Number of total revolutions for spiral pattern',
                        required=False, default=3.0, type=float)
    parser.add_argument('-of','--overlap_fraction', help='Fractional targeted overlap between two adjacent tiles, float between 0 and 1', 
                        required=False, default=cal_fractional_overlap(), type=float)
    parser.add_argument('-ff','--fringe_fraction', help='Fraction of beam radius discarded due to Fresnel fringes, float between 0 and 1', 
                        required=False, default=0, type=float)
    parser.add_argument('-gds','--grouped_dose_symmetric', help='Order for a grouped dose-symmetric (rather than dose-symmetric) scheme',
                        action='store_true')
    parser.add_argument('-o', '--output', help='File name of output',
                        required=True, type=str)
    return vars(parser.parse_args())


def hexagonal_base(args):
    """
    Compute the (x,y) coordinates of the beam centers for hexagonal tiling, ordered such that
    they spiral outwards from the origin to minimize shift required between consecutive tiles.

    Parameters
    ----------
    args : dictionary 
        command-line inputs
    
    Returns
    --------
    ord_beam_pos : numpy.ndarray, shape (n_tiles, 2)
        beam center positions in units of nanometers
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

    Parameters
    ----------
    args : dictionary 
        command-line inputs
    ord_beam_pos : numpy.ndarray, shape (n_tiles, 2)
        beam center positions in units of nanometers
    
    Returns
    --------
    ts_beam_pos : OrderedDict 
        tilt angle: array of beam center coordinates in nanometers
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


def compute_zheights(ts_beam_pos):
    """
    Update each key in the input dictionary to include the estimated z-height 
    at each tile's center.
    
    Parameters
    ----------
    ts_beam_pos : OrderedDict 
        tilt angle: array of beam center coordinates in nanometers
    
    Returns
    --------
    ts_beam_pos : OrderedDict 
        tilt angle: array of beam center coordinates and height in nanometers
    """
    
    for angle in ts_beam_pos.keys():
        z = ts_beam_pos[angle][:,1] * np.tan(np.deg2rad(angle))
        z *= -1 # reverse axis directionality for consistency with SerialEM
        ts_beam_pos[angle] = np.hstack((ts_beam_pos[angle], z[:,np.newaxis]))
        
    return ts_beam_pos


def reorder_grouped_ds(ts_beam_pos, tilt_increment):
    """
    Reorder the tilt angles to follow a grouped dose-symmetric scheme.
    
    Parameters
    ----------
    ts_beam_pos : OrderedDict 
        tilt angle: array of beam center coordinates and height in nanometers
    
    Returns
    --------
    ts_beam_pos : OrderedDict 
        input dictionary, reordered to follow a grouped dose-symmetric scheme
    """
    
    if tilt_increment==2 and len(ts_beam_pos.keys())==61:
        new_order = np.array([ 0,   2,   4,   6,  -2,  -4,  -6,   8,  10,  12,  -8, -10, -12,
                              14,  16,  18, -14, -16, -18,  20,  22,  24, -20, -22, -24,  26,
                              28,  30, -26, -28, -30,  32,  34,  36, -32, -34, -36,  38,  40,
                              42, -38, -40, -42,  44,  46,  48,  50,  52,  54,  56,  58,  60,
                              -44, -46, -48, -50, -52, -54, -56, -58, -60])
        ts_beam_pos = OrderedDict({angle:ts_beam_pos[angle] for angle in new_order})
        
    elif tilt_increment==3 and len(ts_beam_pos.keys())==41:
        new_order = np.array([  0,   3,   6,  -3,  -6,   9,  12,  -9, -12,  15,  18, -15, -18,
                              21,  24, -21, -24,  27,  30, -27, -30,  33,  36, -33, -36,  39,
                              42, -39, -42,  45,  48, -45, -48, -51, -54, -57, -60,  51,  54, 57,  60])
        ts_beam_pos = OrderedDict({angle:ts_beam_pos[angle] for angle in new_order})
       
    else:
        print("Sorry, currently only tilt-acquisition schemes with a tilt increment of\n"+
              "2 or 3 degrees and a tilt-range of -60 to +60 degrees can be reordered \n"+
              "to follow a grouped dose-symmetric pattern. This has to be done manually.")
    
    return ts_beam_pos


def write_coordinates(ts_beam_pos, output, nm_to_um=True):
    """
    Write the data collection coordinates to a text file for SerialEM with 
    the following format: 
    
    angle_0
    x_0 y_0 z_0
    ....
    x_n y_n z_n
    ...
    angle_n
    x_0 y_0 z_0
    ....
    x_n y_n z_n
    
    where there are n tiles collected per tilt angle; the x and y coordinates
    are in the plane of the detector, and the z-coordinate gives the height of
    the tile for auto-focusing. If nm_to_um is True, coordinates are in microns.
    
    Parameters
    ----------
    ts_beam_pos : OrderedDict 
        tilt angle: array of beam center coordinates and height in nanometers
    output : string
        text file for outputting coordinates
    nm_to_um : boolean, default=True
        if True, convert the input coordinates in nanometers to microns
    """
    f = open(output, "w")
    
    scale = 1.0
    if nm_to_um:
        scale = 1.0e-3 
    
    for angle in ts_beam_pos.keys():
        f.write(f"{angle} \n")
        for tile_coords in ts_beam_pos[angle]:
            f.write(' '.join(str(round(val*scale,4)) for val in tile_coords) + '\n')
            
    f.close()
    return


def main():

    args = parse_commandline()
    ord_beam_pos = hexagonal_base(args) # hexagonally-tiled beams
    ts_beam_pos = generate_spiral(args, ord_beam_pos) # spiral for tilt-series
    ts_beam_pos = compute_zheights(ts_beam_pos) # include for defocus estimation
    if args['grouped_dose_symmetric']: # change from dose-symmetric to grouped dose-symmetric
        ts_beam_pos = reorder_grouped_ds(ts_beam_pos, args['tilt_increment'])

    write_coordinates(ts_beam_pos, args['output'])


if __name__ == '__main__':
    main()

import numpy as np

"""
Set default microscope parameters. All lengths are in nm and angles in degrees.
"""

# tilt-scheme information
tilt_angle_increment = 3 # angular increment between tilt images 
tilt_angle_range = [60, -60] # max, min angles of full tilt-range   

# specimen dimensions
vol_size = 40 # voxel size
vol = [5000, 5000, 400] # volume of specimen
rad = 570 # beam radius 

# functions for hexagonal tiling
def hexagonal_tiling(max_one_row, n_interest=7):
    """
    Compute beam center positions (dimensionless) for hexagonally-tiled beams.

    Parameters:
    -----------
    max_one_row: maximum number of tiles along the central row
    n_interest: number of central beams of interest

    Returns:
    --------
    rows: (x,y) coordinates of beam center positions
    act_beam_ind: indices of n_interest central beams of interest
    """
    x_spacing = np.linspace(-int(max_one_row/2)*np.sqrt(3), int(max_one_row/2)*np.sqrt(3), max_one_row)
    row_mid = np.column_stack((x_spacing, np.zeros(len(x_spacing))))
    rows = row_mid

    for i in range(int(max_one_row/2)):
        offset = int(max_one_row/2) - i
        if int(offset/2) == 0: x = x_spacing
        else: x = x_spacing[int(offset/2):-int(offset/2)]
        if offset % 2 != 0:
            row = np.column_stack((x[:-1]+np.sqrt(3)/2, np.zeros(len(x[:-1]))+1.5*offset))        
        else:
            row = np.column_stack((x, np.zeros(len(x))+1.5*offset))
        corr_row = row.copy()
        corr_row[:,1] = -corr_row[:,1]
        rows = np.concatenate((rows, row, corr_row))
        
    # retrieve indices of n_interest central beams
    radii = np.sqrt(np.sum(np.square(rows), axis=1))
    act_beam_ind = np.argsort(radii)[:n_interest]
        
    return rows, np.sort(act_beam_ind)


def cal_fractional_overlap(radius=1, stride=np.sqrt(3), n_overlaps=1):
    """
    Calculate the fraction overlapped area of a circular tile. Formula is based on:
    https://mathworld.wolfram.com/Circle-CircleIntersection.html. Default value 
    is set for perfect hexgonal tiling in which three beams intersect at a point.

    Parameters:
    -----------
    radius: float, beam radius. default=1 (unitless)
    stride: float, distance of beam shift (between centers of adjacent beams)
    n_overlaps: int between 1 and 6, the number of overlaps (lemon shapes) to be considered.
    
    Returns:
    --------
    f_overlap: float, the percentage of overlapped area of a tile
    """

    arc_angle = np.arccos(stride/2/radius) #in radians
    sector_area = (2*arc_angle * radius**2) / 2
    triangle_area = 2*np.sqrt(radius**2 - (stride/2)**2)*(stride/2)/2
    overlap_area = 2*(sector_area - triangle_area) #2 * half lemon shape
    f_overlap = n_overlaps * overlap_area / (np.pi*radius**2)
    return f_overlap


def interpolate_stride(target_overlap, interp_points=1e7, radius=1, n_overlaps=1):
    """
    Given a desired percentage of overlapped area, find the stride by interpolation
    assuming that hexagonal symmetry is maintained along all directions.

    Parameters:
    -----------
    target_overlap: float, desired fraction of overlapped area of a tile
    interp_points: int, number of points for interpolation between stride=radius and stride=2*radius.
    radius: float, beam radius. default=1 (unitless)
    n_overlaps: int between 1 and 6, the number of overlaps (lemon shapes) to be considered.

    Returns:
    --------
    opt_stride: float, approximated stride to reach the target fractional overlap
    """

    stride_arr = np.linspace(radius, 2*radius, int(interp_points))
    calc_overlap = cal_fractional_overlap(stride=stride_arr, radius=radius, n_overlaps=n_overlaps)
    diff = np.abs(calc_overlap-target_overlap)
    return stride_arr[diff.argmin()]

# basic hexagonal tiling: (x,y) coordinates of all beam centers and indices of central seven
max_one_row, n_interest = 9, 7
rand_beam_pos, act_beam_ind = hexagonal_tiling(max_one_row=max_one_row, n_interest=n_interest)
rand_beam_pos *= rad 

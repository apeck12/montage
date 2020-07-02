import numpy as np

"""
Set default microscope parameters. All lengths are in nm and angles in degrees.
"""

# initiaize a hexagonal area with the widest row of max_one_row beams
def hexagonal_tiling(max_one_row, n_interest=7):
    """
    Compute beam center positions (dimensionless) for hexagonally-tiled beams
    with max_one_row tiles setting the maximum number of tiles along any row. 
    Also return indices of central n_interest beams of interest.
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

## tilt-scheme information 
tilt_angle_increment = 3 # angular increment between tilt images 
tilt_angle_range = [60, -60] # max, min angles of full tilt-range   

## specimen dimensions
vol_size = 40 # voxel size
vol = [5000, 5000, 400] # volume of specimen
rad = 570 # beam radius 

"""
the commented section is moved to prepare_beam() in simulate_spiral.py to accomodate changes in 
overlaps of beams.
"""
# # beam information: (x,y) coordinates of all beam centers and indices of central seven
# rand_beam_pos, act_beam_ind = hexagonal_tiling(max_one_row=7, n_interest=7)
# rand_beam_pos *= rad

"""
Calculate the percentage of overlapped area of a tile. Formula is based on:
https://mathworld.wolfram.com/Circle-CircleIntersection.html
Default value is set for perfect hexgonal tiling with no additional overlaps.

Inputs:
-------
radius: float, beam radius.
stride: float, distance of beam shift. Only the ratio of stride:radius matters.
n_overlaps: int between 1 and 6, the number of overlaps (lemon shapes) to be considered.

Ouputs:
-------
overlap_percent: float, the percentage of overlapped area of a tile.
""" 
def cal_overlap_percent(radius=1, stride=np.sqrt(3), n_overlaps=1):
    arc_angle = np.arccos(stride/2/radius) #in radians
    sector_area = (2*arc_angle * radius**2) / 2
    triangle_area = 2*np.sqrt(radius**2 - (stride/2)**2)*(stride/2)/2
    overlap_area = 2*(sector_area - triangle_area) #2 * half lemon shape
    overlap_percent = n_overlaps * overlap_area / (np.pi*radius**2)
    return overlap_percent

"""
Given a desired percentage of overlapped area, find the stride by interpolation (analytical formula
is hard to write), assuming hexagons maintained (i.e. identical strides along all 6 directions).

Inputs:
-------
target_overlap_pct: float, desired percentage of overlapped area of a tile
interp_points: int, number of points for interpolation between stride=radius and stride=2*radius.
radius: float, beam radius.
n_overlaps: int between 1 and 6, the number of overlaps (lemon shapes) to be considered.

Ouputs:
-------
opt_stride: float, approximated stride to reach the target overlap percentage.
"""
def interpolate_stride(target_overlap_pct, interp_points=1e7, radius=1, n_overlaps=1):
    stride_arr = np.linspace(radius, 2*radius, interp_points)
    overlap_arr = cal_overlap_percent(stride=stride_arr, radius=radius, n_overlaps=n_overlaps)
    diff = np.abs(overlap_arr-target_overlap_pct)
    return stride_arr[diff.argmin()]
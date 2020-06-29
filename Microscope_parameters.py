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

# beam information: (x,y) coordinates of all beam centers and indices of central seven
rand_beam_pos, act_beam_ind = hexagonal_tiling(max_one_row=7, n_interest=7)
rand_beam_pos *= rad


import numpy as np
import mrcfile

def apply_rotation(beam_centers, rotation_angle):
    """
    Rotate beam centers in plane of detector.
    
    Parameters
    ----------
    beam_centers : numpy.ndaarray, shape (N, 2)
        beam centers
    rotation_angle : float 
        rotation angle to apply in degrees
    
    Returns
    -------
    r_beam_centers : numpy.ndarray, shape (N, 2) 
        rotated beam centers
    """
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    return beam_centers.dot(R)


def save_mrc(data, savename, voxel_size=None):
    """
    Save data in .mrc format.
    
    Parameters
    ----------
    data : numpy.ndarray, 2d or 3d
        image or volume to save
    savename : string
        output file path
    voxel_size: float 
        voxel size for header in Angstrom, optional
    """
    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    if voxel_size is not None:
        mrc.voxel_size = voxel_size
    mrc.close()
    return

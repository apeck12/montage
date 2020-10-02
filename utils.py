import numpy as np
import numpy.ma as ma
import mrcfile

"""
A collection of functions that are used repeatedly during data processing.
"""

######################################
# Functions for image I/O operations #
######################################

def load_mask_tile(tile_path, mask_path, as_masked_array=True):
    """
    Load tile and mask, optionally converting to a numpy masked array.
    
    Inputs:
    -------
    tile_path: path to tile in MRC format
    mask_path: path to mask in npy format, value of 0 means discard
    as_masked_array: boolean dictating whether to convert to masked array
    
    Outputs:
    --------
    m_tile: tile in masked array format (if as_masked_array)
    tile, mask: tile and mask as separate numpy arrays (if not as_masked_array)
    """
    # load image file
    mrc_tile = mrcfile.open(tile_path)
    tile = mrc_tile.data.copy()
    mrc_tile.close()
    
    # load mask file; value of 0 corresponds to masked region
    if mask_path[-3:] == 'mrc':
        mask = mrcfile.open(mask_path).data.astype(bool)
    else:
        mask = np.load(mask_path).astype(bool)
    
    # optionally convert to masked array format
    if as_masked_array is False:
        return tile, mask
    else:
        return ma.masked_array(tile, mask=np.invert(mask))


def save_mrc(data, savename):
    """
    Save Nd numpy array to path savename in mrc format.
    
    Inputs:
    -------
    data: Nd array to be saved
    savename: path to which to save Nd array in mrc format
    """
    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(data.astype(np.float32))
    mrc.close()
    return


###################################################################
# Image processing functions: normalization, transformation, etc. #
###################################################################  


def normalize(tile, mu=None, sigma=None):
    """
    Standardize values of tile to have a mean of 0 and a standard deviation of 1:
    norm_image = (image - image.mean()) / image.std()
    Alternatively, mean and standard deviation can be supplied to normalize tiles
    based on statistics for all tiles from the tilt angle. It's assumed that the 
    image is supplied as a masked array.
    
    Inputs:
    -------
    tile: tile in masked array format
    mu: global mean to use for standardization, optional
    sigma: global standard deviation to use for standardization, optional
    
    Outputs:
    --------
    tile: normalized tile in masked array format
    """
    if mu is None: mu = np.mean(tile)
    if sigma is None: sigma = np.std(tile)
        
    return (tile - mu) / sigma


def apply_affine(image, mask, M, offset=np.zeros(2)):
    """
    Apply an affine transformation to the image and mask, generate a masked array
    from the transformed result, and normalize.
    
    Inputs:
    -------
    image: 2d array corresponding to image
    mask: 2d array corresponding to mask
    M: affine transformation matrix
    offset: translational offset, default is no shift
    
    Outputs:
    --------
    tile: normalized, affine-transformed tile as a masked array
    """
    import scipy.ndimage

    tile = scipy.ndimage.interpolation.affine_transform(image, M, offset)
    mask_t = scipy.ndimage.interpolation.affine_transform(mask.astype(int), M, offset)
    tile = ma.masked_array(tile, mask=np.invert(np.around(mask_t).astype(int).astype(bool)))
    tile = normalize(tile)
    
    return tile

#########################################################
# 2D transformation matrices for affine transformations #
######################################################### 

def rotation_matrix(theta):
    """
    Compute 2d matrix for counter-clockwise rotation about origin by theta.
    
    Inputs:
    -------
    theta: angle in radians
    
    Outputs:
    --------
    R: rotation matrix
    """
    return np.array([[np.cos(theta),-1*np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])


def scaling_matrix(s_x, s_y):
    """
    Compute 2d matrix for scaling transformation.
    
    Inputs:
    -------
    s_x: scale parameter along x axis
    s_y: scale parameter along y axis
    
    Outputs:
    --------
    S: scaling matrix
    """
    return np.array([[s_x,0],[0,s_y]]) 


def shearing_matrix(sh_x, sh_y):
    """
    Compute 2d matrix for shearing transformation.
    
    Inputs:
    -------
    sh_x: shear parameter parallel to x axis
    sh_y: shear parameter parallel to y axis
    
    Outputs:
    --------
    Sh: shearing matrix
    """
    return np.array([[1.0,sh_x],[sh_y,1.0]])



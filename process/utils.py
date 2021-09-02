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

def patch_image(region, sigma=0.19):
    """
    Patch zero-valued gap between tiles in a subregion of an image. 
    The gap is identified by applying tools from scipy.ndimage to the
    Gaussian-filtered subregion, and then sampling from the remaining
    pixels in the subregion to fill the gap.
    
    Parameters
    ----------
    region : numpy.ndarray, shape (M,N)
        subimage of full stitch to patch
    sigma : float
        kernel size for Gaussian filter
        
    Returns
    -------
    region : numpy.ndarray, shape (M,N)
        patched subimage
    """
    import scipy.ndimage
    
    region_filt = scipy.ndimage.gaussian_filter(region, sigma)

    mask = np.zeros_like(region)
    mask[region_filt==0] = 1

    struct = scipy.ndimage.generate_binary_structure(2, 1)
    labeled, ncomponents = scipy.ndimage.measurements.label(mask, struct)
    
    region[labeled!=0] = np.random.choice(region[labeled==0], size=region[labeled!=0].shape[0])
    
    return region

def patch_region_inside_vertices(region, vertices):
    """
    Patch the portion of the region that lies inside the given vertices.
    
    Paramters
    ---------
    region : numpy.ndarray, shape (M,N)
        subimage of full stitch to patch
    vertices : numpy.ndarray, shape (P,2)
        vertices of polygon that encloses the area to patch
        
    Returns
    -------
    region : numpy.ndarray, shape (M,N)
        patched subimage
    """
    import matplotlib.path
    
    x,y = np.arange(region.shape[1]), np.arange(region.shape[0])
    x,y = np.meshgrid(x,y)
    positions = np.vstack([x.ravel(), y.ravel()]).T

    p = matplotlib.path.Path(vertices)
    interior = p.contains_points(positions).reshape(region.shape)
    region[interior] = np.random.choice(region[~interior], size=region[interior].shape[0])
    
    return region

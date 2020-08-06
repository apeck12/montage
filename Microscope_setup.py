from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
from scipy import spatial
import numpy as np
import os, itertools, time
from collections import OrderedDict
from pathos.multiprocessing import ProcessingPool
import pathos.pools as pp


class Sample:
    
    """
    Class for discretizing the sample, determining which sample voxels fall within the 
    region of interest, and keeping track of exposure counts. The region of interest is
    either those voxels spanned by a subset of the beam tiles or a rectangular region on 
    the untilted specimen that is constant throughout the depth of the sample.
    """
    
    def __init__(self, volume_3d, voxel_size, angle=0, interested_area=None):
        """
        Initialize instance of class, including calculation of voxel centers for
        discretized sample. 
        """
        self.x_len, self.y_len, self.z_len = tuple(volume_3d) # (x,y,z) dimensions of sample in nm
        self.voxel_size = voxel_size # length of cubic voxel in nm
        self.angle = angle # current angle of tilt-series, initially set to 0 (normal to beam)
        self.__ori_center__ = np.array(volume_3d)/2.0 # set origin to sample center
        self.interest_mask = None
        self.vx, self.vy, self.vz = self.__get_voxel_centers__() # x,y,z positions of voxel centers
        self.interested_area = interested_area # either 'all_circles' or string of 'x,y' dimensions
        
    def __get_voxel_centers__(self):
        """
        Compute the (x,y,z) position of each voxel in the sample. The coordinate system has its
        origin in the center of the sample, and the sample volume is discretized based on the
        dimensions of volume_3d used to set up class and self.voxel_size. The sample is untilted.
        
        Outputs:
        --------
        vox_centers_x, vox_centers_y, vox_centers_z: arrays of coordinate positions along x/y/z
            axes, respectively, of sample's voxel centers
        """
        # compute the (x,y,z) coordinates of all voxels in discretized sample
        vox_centers_x = np.linspace(self.voxel_size/2, self.x_len-self.voxel_size/2, 
                                    int(self.x_len/self.voxel_size)) - self.__ori_center__[0]
        vox_centers_y = np.linspace(self.voxel_size/2, self.y_len-self.voxel_size/2, 
                                    int(self.y_len/self.voxel_size)) - self.__ori_center__[1]
        vox_centers_z = np.linspace(self.voxel_size/2, self.z_len-self.voxel_size/2, 
                                    int(self.z_len/self.voxel_size)) - self.__ori_center__[2]
        self.voxel_centers = np.array(list(itertools.product(vox_centers_x, vox_centers_y,
                                                             vox_centers_z)))
        
        # set additional class variables: arrays for num counts and voxels in region of interest
        self.exposure_counter = np.zeros(len(self.voxel_centers), dtype=int)
        self.interest_mask = np.ones(len(self.voxel_centers), dtype=bool)

        return (vox_centers_x, vox_centers_y, vox_centers_z)
    
    def __set_rectangular_area__(self):
        """
        Helper function for setting a rectangular prism as the area of interest of the sample.
        """
        #check valid input
        try:
            self.interested_area = np.array(self.interested_area.split('_'),dtype=float)
            assert(len(self.interested_area) == 2)
        except ValueError as error1:
            print('Requires float type for setting interested area of the sample')
            raise error1
        except AssertionError as error2:
            print('Requires two floats for setting interested area of the sample')
            raise error2
        # if the number of voxels is odd relative to the entire discretized specimen, the additional
        # voxel is added to the right/positive side: e.g. x_masks = [0 0 1 1 1 0] or [0 0 1 1 1 1 0]
        num_sample = np.array([len(self.vx), len(self.vy)])
        num_area = np.array(self.interested_area/self.voxel_size, dtype=int)
        odd_voxels = np.array((num_sample - num_area)%2, dtype=int)
        # if size not multiple of vs, it is down calculated: e.g. with voxel size=2nm, specifying
        # 5nm length is the same as 4nm.
        half_num_voxels = np.array(np.array(self.interested_area)/self.voxel_size/2, dtype=int)
        x_ind = int(len(self.vx)/2) - half_num_voxels[0]
        y_ind = int(len(self.vy)/2) - half_num_voxels[1]

        x_masks = np.zeros(len(self.vx))
        if len(self.vx) % 2 == 0: 
            start=x_ind
            end=-x_ind+odd_voxels[0]
        else: 
            start=x_ind+odd_voxels[0]
            end=-x_ind
        if end == 0: x_masks[start:] = 1
        else: x_masks[start:end] = 1

        y_masks = np.zeros(len(self.vy))
        if len(self.vy) % 2 == 0: 
            start=y_ind
            end=-y_ind+odd_voxels[1]
        else: 
            start=y_ind+odd_voxels[1]
            end=-y_ind
        if end == 0: y_masks[start:] = 1
        else: y_masks[start:end] = 1

        z_masks = np.ones(len(self.vz))
        mask = np.array(list(itertools.product(x_masks, y_masks, z_masks)))
        self.interest_mask = mask.prod(axis=1) == 1
    
    def set_interested_area(self, beam=None, voxel_dir=None):
        """
        Generate a mask (self.interest_mask) that corresponds to the sample voxels in the area
        of interest.
        If self.interested_area is 'all_circles', the region of interest includes all sample 
        voxels spanned by the act_beam_ind tiles of the Beam object. 
        If self.interested_area is 'num1_num2', the region of interest is a rectangular prism
        centered at the origin of size num1(x-axis) and num2(y-axis) in unit of nm.
        If, self.interested_area is 'num', the region of interest is a cylinder centered at
        the origin of radius num in unit of nm.
        
        Inputs (optional):
        -------
        beam: instance of Beam class, with beam.patterns defined. only needed for 'all_circles'.
        voxel_dir: path to pre-computed voxel coordinates arrays. only needed for 'all_circles'.
        """
        # set interest_mask region to be act_beam_ind tiles of Beam object
        if self.interested_area == 'all_circles':
            #all_tiles: array of exposure counts for untilted sample and beam pattern at 0 degrees
            all_tiles = beam.image_all_beams(voxel_dir, 0, self.voxel_size, sample_init=True)
            self.interest_mask = all_tiles >= 1 #boolean version of all_tiles
            
        # set interest_mask region to a rectangular region, an example input is '100.0_90.0'
        # meaning the interest area is the CENTER rectangular of 100.0nm(x) and 90.0nm(y) size.
        elif '_' in self.interested_area:
            self.__set_rectangular_area__()
        
        # set interest_mask region to the CENTER circle/cylinder region of radius specified
        # by user input in unit of nm.
        else:
            try: self.interested_area = float(self.interested_area)
            except ValueError as error:
                print('Requires float type for setting interested area of the sample')
                raise error
            center = [0,0]
            dists = distance_matrix(self.voxel_centers[:,:-1], [center])[:,0] #[:, 0] for flattening
            self.interest_mask = dists <= self.interested_area
        return
       

class SampleHolder:

    """
    Class for tilting discretized sample to each angle in the tilt-series.
    """
    
    def __init__(self, sample, tilt_range=[60, -60], tilt_increment=3, tilt_axis='x'):
        """
        Initialize instance of SampleHolder class. The angles to be tilted to, excluding 
        the current angle, are updated and stored in self.all_tilt_angles.
        
        Inputs:
        -------
        sample: instance of Sample class 
        tilt_range: np.array of [max,min] angles of tilt-series, inclusive, in degrees
        tilt_increment: angular increment between tilt images in degrees
        tilt_axis: string indicating which single axis about which to tilt sample
        """
        self.tilt_range = tilt_range 
        self.tilt_increment = tilt_increment 
        self.tilt_axis = tilt_axis 
        self.all_tilt_angles = self.get_remain_tilt_angles(sample)  
    
    def get_remain_tilt_angles(self, sample):
        """
        Retriieve remaining angles in tilt-series, assuming that the sample has not yet
        been imaged at the current angle. 
        
        Inputs:
        -------
        sample: instance of Sample class
        
        Outputs:
        --------
        angles: array of angles in tilt-series
        """
        angles = [sample.angle]
        while True:
            self.__raptor_set_angle__(sample)
            if angles[-1] == sample.angle: break
            angles.append(sample.angle)
        sample.angle = angles[0] # set it back to the initial angle, since raptor_tilt updates it
        return np.array(angles)
    
    def __raptor_set_angle__(self, sample): 
        """
        Update sample.angle by performing one tilt in a raptor (dose-symmetric) pattern.
        """
        if sample.angle > 0:
            if -sample.angle >= self.tilt_range[1]: 
                sample.angle = -sample.angle
            elif sample.angle + self.tilt_increment <= self.tilt_range[0]: 
                sample.angle += self.tilt_increment
        elif sample.angle <= 0:
            if -sample.angle + self.tilt_increment <= self.tilt_range[0]: 
                sample.angle = -sample.angle + self.tilt_increment
            elif sample.angle - self.tilt_increment >= self.tilt_range[0]: 
                sample.angle -= self.tilt_increment
        return
                
    def __rotate_sample__(self, sample, sample_mask, angle):
        """
        Rotate subset of sample voxels defined by boolean array sample_mask by the input
        angle around self.tilt_axis.
        
        Inputs:
        -------
        sample: instance of Sample class
        sample_mask: boolean array of shape sample
        angle: angle by which to rotate sample subset around self.tilt_axis, in degrees
        """
        Rot_matrix = R.from_euler(self.tilt_axis, angle, degrees=True)
        sample.voxel_centers[sample_mask] = Rot_matrix.apply(sample.voxel_centers[sample_mask])
        return
    
    def __raptor_rotate_coord__(self, prev_angle, sample):
        """
        Update the coordinates of the sample voxels to the next tilt increment.
        """
        self.__rotate_sample__(sample, sample.interest_mask, sample.angle-prev_angle)
        return
        
    def raptor_tilt(self, sample):
        """
        Perform one tilt on sample, updating both sample.angle and sample.voxel_centers.
        """
        prev_angle = sample.angle
        self.__raptor_set_angle__(sample)
        self.__raptor_rotate_coord__(prev_angle, sample)
        return
        

class Beam:
    
    """
    Class for setting up beam positions and computing which specimen voxels are imaged by 
    those beams at each angle of the tilt-series.
    """

    def __init__(self, radius, beam_pos=[(0,0)], actual_beam_ind=0, n_processor=4):
        """
        Initialize instance of class, including generation of basic spiral pattern.
        Note that either alternate or continuous arguments must be True.
        """
        self.radius = radius # radius of beam in nm, float
        self.n_processes = n_processor # number of CPU processors, int
        self.__all_pos__ = np.array(beam_pos) # 2d array of (x,y) hexagonally-tiled centers
        self.actual_beams = np.zeros(self.__all_pos__.shape[0], dtype=bool) 
        self.actual_beams[actual_beam_ind] = True # set which beams are of interest
        self.patterns = None # for storing beam patterns, see Beam_offset_generator class
        
    def image_all_beams(self, voxel_dir, curr_angle, vox_size, sample_init=False):
        """
        Compute the number of times each voxel in the specimen is imaged at the specified
        tilt angle. If sample_init is True, then only consider a subset of beams dictated
        by self.actual_beams; otherwise, consider all beam positions.
        
        Inputs:
        -------
        voxel_dir: path to pre-computed voxel coordinates arrays
        curr_angle: angle of tilt-series being processed
        vox_size: length of cubic voxel in nm
        
        Outputs:
        --------
        mask: 1d array of how many times each voxel is imaged for curr_angle beam positions
        """

        #update beam positions by some offset
        self.__all_pos__ = self.patterns[curr_angle]
        
        # consider either subset of beams or all beams
        if sample_init: pos = self.__all_pos__[self.actual_beams] 
        else: pos = self.__all_pos__
        
        voxel_file = os.path.join(voxel_dir, f'vs{vox_size}nm_{curr_angle}.npy')
        sample_voxels = np.load(voxel_file, mmap_mode='r')
        
        # for each beam tile, compute which voxels are imaged
        mask = np.zeros(sample_voxels.shape[0], dtype=int)
        for p in pos:
            dists = distance_matrix(sample_voxels, [p])[:,0]
            mask_one_beam = dists <= self.radius
            mask += mask_one_beam

        return mask

    def __image_all_beams_wrapper__(self, args):
        """
        Wrapper for passing along pickled arguments for multiprocessing.
        """
        return self.image_all_beams(*args)
    
    def image_all_tilts(self, voxel_dir, all_angles, vox_size):
        """
        Image all angles in tilt-series, parallelizing the calculation of which voxels
        are imaged by a beam using pathos multiprocessing.
        
        Inputs:
        -------
        voxel_dir: path to pre-computed voxel coordinates arrays
        all_angles: array of angles in tilt-series
        vox_size: length of cubic voxel in nm
        
        Outputs:
        --------
        all_masks: 1d array of number of times each voxel has been imaged, 
            shape of sample.interest_mask
        """
        pool = pp.ProcessPool(self.n_processes)
        all_masks = sum(pool.map(self.__image_all_beams_wrapper__, 
                                 zip([voxel_dir]*len(all_angles), all_angles, 
                                     [vox_size]*len(all_angles))))
        return all_masks         
                                 

class Beam_offset_generator_snowflake:

    """
    Class for computing beam center positions that are hexagonally-tiled on individual
    and follow a snowflake pattern over the course of the tilt-series.
    """

    def __init__(self, radius, beam_positions, tilt_series, 
                 n_steps=2, max_trans=2, xscale=False, 
                 start_beam_angle=0, rotation_step=30, alternate=False):
        """
        Initialize instance of class, including generation of basic snowflake pattern.
        """
        self.radius = radius # radius of beam in nanometers
        self.beam_pos = beam_positions # array of hexagonally-tiled beam centers
        self.tilt_series = tilt_series # array of ordered tilt angles
        self.snowflake = self.__generate_snowflake__(n_steps, max_trans)
        self.Xscale = xscale # boolean: scale the X-coordinate of the translation
        self.start_beam_angle = start_beam_angle # initial angular offset in degrees
        self.rotation_step = rotation_step # angular reorientation of spiral between images
        self.alternate = alternate # boolean: alternate between snowflake pattern offset by rotation_step
        self.offset_patterns = OrderedDict()
        
    def __generate_snowflake__(self, n_steps, max_trans):
        """
        Generate basic snowflake pattern, where each coordinate corresponds to one of 
        the positions the center of the hexagonally-tiled pattern will be shifted to.
        
        Inputs:
        -------
        n_steps: number of positions between origin and outermost edge of snowflake
        max_trans: maximum radial distance (units: radius) permitted for a position
        
        Outputs:
        --------
        snowflake: 2d array of (x,y) positions on the snowflake
        """
        # condition of zero translation
        if (n_steps == 0) or (max_trans==0):
            snow_pat = np.array([[0,0]])
        
        else:
            # generate positions along each of the three axes, both positive and negative
            three_axes = np.array([[0,1],[np.sqrt(3)/2,0.5],[np.sqrt(3)/2,-0.5]]) 
            xy_pos = np.linspace(0,three_axes,n_steps+1)[1:].reshape(3*(n_steps),2)
            xy_neg = -1 * xy_pos

            # ordering is such that origin is first, then spiraling clockwise outwards
            snow_pat = np.array([0,0])
            for i in range(int(xy_pos.shape[0]/3)):
                snow_pat = np.vstack((snow_pat, xy_pos[i*3:(i*3)+3]))
                snow_pat = np.vstack((snow_pat, xy_neg[i*3:(i*3)+3]))
            snow_pat = snow_pat * max_trans * self.radius
            
        return snow_pat

    def offset_all_beams(self, file_path=None):
        """
        For each tilt angle, offset the original hexagonally-tiled beam positions
        based on the specified translational (snowflake) and rotational elements.
        
        Inputs:
        -------
        file_path: path to store pickled beam centers, optional
        """
        
        for a, tilt_angle in enumerate(self.tilt_series):
            # set rotational component: optional rotation between tilts and angular offset
            theta = 0
            if a == 0: 
                theta = np.radians(self.start_beam_angle)
            elif self.alternate:
                if a%2 == 1: theta = np.radians(self.rotation_step)
                else: theta = np.radians(-self.rotation_step)
            else:
                if a%(2*len(self.snowflake)) == 0: theta = np.radians(-self.rotation_step)
                elif a%len(self.snowflake) == 0: theta = np.radians(self.rotation_step)
            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array(((c, -s), (s, c)))
            self.beam_pos = Rx.dot(self.beam_pos.T).T
            self.snowflake = Rx.dot(self.snowflake.T).T
             
            # translational element: ath element of spiral, optionally X-scaled
            translation = self.snowflake[a%len(self.snowflake)].copy()
            if self.Xscale:
                translation[0] /= np.cos(np.deg2rad(tilt_angle))
                abs_offset = np.abs(translation[0])%(3*self.radius)
                if translation[0] >= 0: translation[0] = abs_offset
                else: translation[0] = -abs_offset
                
            self.offset_patterns[tilt_angle] = self.beam_pos + translation
            
        # if a file path is supplied, save the beam positions
        if file_path is not None:
            with open(file_path, 'wb') as handle:
                pickle.dump(self.offset_patterns, handle)

        return
            

class Beam_offset_generator_spiral:
    
    """
    Class for computing beam center positions that are hexagonally-tiled on individual
    tilt images and follow a spiral pattern over the course of the tilt-series.
    """
    
    def __init__(self, radius, beam_positions, tilt_series, n_revolutions, max_trans=2, 
                 xscale=False, start_beam_angle=0, rotation_step=30, alternate=False,
                 continuous=False):
        """
        Initialize instance of class, including generation of basic spiral pattern.
        Note that either alternate or continuous arguments must be True.
        """
        self.radius = radius # radius of beam
        self.beam_pos = beam_positions # array of hexagonally-tiled beam centers
        self.tilt_series = tilt_series # array of ordered tilt angles
        self.spiral = self.__generate_spiral__(n_revolutions, max_trans)
        self.Xscale = xscale # boolean: scale the X-coordinate of the translation
        self.start_beam_angle = start_beam_angle # initial angular offset in degrees
        self.rotation_step = rotation_step # angular reorientation of spiral between images
        self.alternate = alternate # boolean: alternate rotation between images
        self.continuous = continuous # boolean: rotate by rotation_step between images
        self.offset_patterns = OrderedDict()
            
    def __generate_spiral__(self, n_revolutions, max_translation):
        """
        Generate equidistant points along an Archimedean spiral, equations
        courtesy: https://downloads.imagej.net/fiji/snapshots/arc_length.pdf.
        Number of points is the same as number of projection images; points
        spiral outwards from the center / origin.

        Inputs:
        -------
        n_revolutions: approximate number of revolutions for entire spiral
        max_translation: maximum distance from origin any point is allowed to take

        Outputs:
        --------
        spiral: (x,y) coordinates along spiral, np.array of shape (n_tilts, 2)
        """
        # for the special case of 0 revolutions, introduce no translational offset
        if n_revolutions == 0:
            return np.zeros((len(self.tilt_series),2))
        
        else:
            a, b = 0.1, 10.0 # good default parameters; these eq. are an approximation
            s = np.arange(len(self.tilt_series))
            factor = b * n_revolutions * np.pi / (2 * np.max(s))
            thetas = factor * 2*np.pi * np.sqrt(2*s/b)
            x = (a + b*thetas) * np.cos(thetas)
            y = (a + b*thetas) * np.sin(thetas)
            x[0],y[0] = 0,0 # eliminate rounding errors for center of spiral

            # make sure that no point is farther than max_translation from origin
            max_dist = np.max(np.sqrt(np.square(x) + np.square(y))) / max_translation
            x /= max_dist
            y /= max_dist

            return np.array([x,y]).T * self.radius
    
    def offset_all_beams(self, file_path=None, max_offset=3):
        """
        For each tilt angle, offset the original hexagonally-tiled beam positions
        based on the specified translational (spiral) and rotational elements.
        
        Inputs:
        -------
        file_path: path to store pickled beam centers, optional
        max_offset: threshold in number of radii for maximum Xscale translation
        """
        for a,tilt_angle in enumerate(self.tilt_series):
            # rotational element: reorient hexagonally-tiled centers
            if self.alternate:
                if a%2 == 1: theta = np.radians(self.rotation_step)
                else: theta = np.radians(-self.rotation_step)
            elif self.continuous: # overriden if alternate is True
                theta = a*np.radians(self.rotation_step)
            theta += np.radians(self.start_beam_angle)

            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array(((c, -s), (s, c)))
            beam_pos_r = Rx.dot(self.beam_pos.copy().T).T

            # translational element: ath element of spiral, opt. X-scaled
            translation = self.spiral[a].copy()
            if self.Xscale:
                translation[0] /= np.cos(np.deg2rad(tilt_angle))
                abs_offset = np.abs(translation[0])%(max_offset*self.radius)
                if translation[0] >= 0: translation[0] = abs_offset
                else: translation[0] = -abs_offset

            self.offset_patterns[tilt_angle] = beam_pos_r + translation
          
        if file_path is not None:
            with open(file_path, 'wb') as handle:
                pickle.dump(self.offset_patterns, handle)
                
        return


class Beam_offset_generator_sunflower:
    
    """
    Class for computing beam center positions that are hexagonally-tiled on individual
    tilt images and follow a sunflower pattern over the course of the tilt-series. The 
    sunflower is a generative spiral that evenly distributes points in a 2d disc; see:
    Swinbank and Purser (2006) Quarterly Journal of the Royal Meteorological Society. 
    """
    
    def __init__(self, radius, beam_positions, tilt_series, max_trans=2, xscale=False, 
                 start_beam_angle=0, rotation_step=30, alternate=True, continuous=False):
        """
        Initialize instance of class, including generation of basic sunflower pattern.
        Note that either alternate or continuous arguments must be True.
        """
        self.radius = radius # radius of beam
        self.beam_pos = beam_positions # array of hexagonally-tiled beam centers
        self.tilt_series = tilt_series # array of ordered tilt angles
        self.sunflower = self.__generate_sunflower__(max_trans)
        self.Xscale = xscale # boolean: scale the X-coordinate of the translation
        self.start_beam_angle = start_beam_angle # initial angular offset in degrees
        self.rotation_step = rotation_step # angular reorientation of spiral between images
        self.alternate = alternate # boolean: alternate rotation between images
        self.continuous = continuous # boolean: rotate by rotation_step between images
        self.offset_patterns = OrderedDict()
            
    def __generate_sunflower__(self, max_translation):
        """
        Generate the base sunflower pattern of evenly distributed points in a circle,
        where each (x,y) coordinate corresponds to the position of the central tile.
        See reference above for equations.

        Inputs:
        -------
        max_translation: maximum distance from origin any point is allowed to take
        
        Outputs:
        --------
        sunflower: (x,y) coordinates along sunflower, np.array of shape (n_tilts, 2)
        """
        # for the special case of 0 max_trans, introduce no translational offset
        if max_translation == 0:
            return np.zeros((len(self.tilt_series),2))
        
        else:
            n = np.arange(len(self.tilt_series))+1
            r = np.sqrt(n/float(len(n)))
            theta = 2*np.pi*n*(0.5*(1+np.sqrt(5))-1)
            x,y = r*np.cos(theta), r*np.sin(theta)

            x *= max_translation
            y *= max_translation
            return np.array([x,y]).T * self.radius    

    def offset_all_beams(self, file_path=None, max_offset=3):
        """
        For each tilt angle, offset the original hexagonally-tiled beam positions
        based on the specified translational (sunflower) and rotational elements.
        
        Inputs:
        -------
        file_path: path to store pickled beam centers, optional
        max_offset: threshold in number of radii for maximum Xscale translation
        """
        for a,tilt_angle in enumerate(self.tilt_series):
            # rotational element: reorient hexagonally-tiled centers
            if self.alternate:
                if a%2 == 1: theta = np.radians(self.rotation_step)
                else: theta = np.radians(-self.rotation_step)
            elif self.continuous: # overriden if alternate is True
                theta = a*np.radians(self.rotation_step)
            theta += np.radians(self.start_beam_angle)

            c, s = np.cos(theta), np.sin(theta)
            Rx = np.array(((c, -s), (s, c)))
            beam_pos_r = Rx.dot(self.beam_pos.copy().T).T

            # translational element: ath element of sunflower, opt. X-scaled
            translation = self.sunflower[a].copy()
            if self.Xscale:
                translation[0] /= np.cos(np.deg2rad(tilt_angle))
                abs_offset = np.abs(translation[0])%(max_offset*self.radius)
                if translation[0] >= 0: translation[0] = abs_offset
                else: translation[0] = -abs_offset

            self.offset_patterns[tilt_angle] = beam_pos_r + translation
          
        if file_path is not None:
            with open(file_path, 'wb') as handle:
                pickle.dump(self.offset_patterns, handle)
                
        return

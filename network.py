import itertools
import numpy as np
import scipy as sp

class Connectivity:
    def __init__(self, filters):
        '''
        Class for building a connectivity matrix.
        Args:
            filters = convolution filters for angles detection
        '''
        self.filters = filters
    
    def spatial_connect(self, width, height):
        '''
        Calculating spatial connectivity between all neurons.
        '''
        neurons_coord = np.array(list(itertools.product(range(height), range(width))))
        distances = sp.spatial.distance.cdist(neurons_coord, neurons_coord, 'chebyshev')
        distances = np.tile(distances, (len(self.filters),len(self.filters)))
        return 1/(distances+1)
    
    def angle_connect(self, width, height):
        '''
        Calculating angular connectivity between all neurons.
        '''
        def angle_diff(x,y):
            x,y = sorted([x,y])
            abs_diff = 180 - abs(abs(x-y) - 180)
            if (x == 0 and y > 90) or (x < 90 and y == 180):
                return (180 - abs_diff)/45
            return abs_diff/45   
        
        angles = [180//4*i for i in range(len(self.filters))]
        vec_len = width * height
        angle_diffs = np.zeros((vec_len*len(self.filters), vec_len*len(self.filters)))
        
        for i in range(len(angles)):
            for j in range(len(angles)):
                angle_diffs[vec_len*i:vec_len*(i+1),vec_len*j:vec_len*(j+1)] = angle_diff(angles[i], angles[j])
                
        return 1/(angle_diffs+1)
    
    def build(self, width, height, angle_connect_strength=0.5, spatial_connect_strength=0.5, total_connect_strength=1):
        '''
        Build a connection matrix which takes into account both spatial distance and angular difference between all pairs of neurons.
        '''
        return (self.angle_connect(width, height) * angle_connect_strength + self.spatial_connect(width, height) * spatial_connect_strength) * total_connect_strength
    
    def detect_angles(self, img, step=1, width_start=0, height_start=0):
        '''
        Detect angles of the input image wih use of the convolutional filters.
        Args:
            img = input image
        '''
        filtered = np.zeros((len(self.filters), img.shape[0], img.shape[1]))

        for i, f in enumerate(self.filters):
            filtered[i] = sp.signal.convolve(img, f, mode='same', method='direct')
        
        return filtered[:, height_start::step, height_start::step]
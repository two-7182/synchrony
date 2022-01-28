import numpy as np
import scipy as sp
import time
from network import Connectivity
from model import Izhikevich

class Simulation:
    def __init__(self, stimulus, filters, filter_step=1, filter_start_width=0, filter_start_height=0, decay=0.1, angle_connect_strength=0.5, spatial_connect_strength=0.5, thresh=30, total_connect_strength=0.5):
        """
        It builds the connections network and runs the simulation of a given length.
        Args:
            stimulus = the input stimulus image
            filters = convolution filters for angles recognition
            filter_step = step of the convolution filters
            filter_start_width = the starting x point of the convolution filters
            filter_start_height = the starting y point of the convolution filters
            decay = strength of decay
            angle_connect_strength = the strength of angular connections between neurons
            spatial_connect_strength = the strength of spatial connections between neurons
            total_connect_strength = the strength of final connections between neurons
        """
        
        if len(stimulus.shape) != 2:
            raise Exception("Please use a 2-dimensional stimulus image")
        
        #initialization of the Connectivity class with filters
        self.connectivity = Connectivity(filters)
        
        #input preprocessing
        self.neural_input = self._preprocess_input(stimulus, filter_step, filter_start_width, filter_start_height)
        
        #initialization of helper simulation parameters
        self.network_shape = self.neural_input.shape
        self.n_neurons = np.prod(self.neural_input.shape)
        
        #transform array 
        self.neural_input = self.neural_input.reshape(-1)
        
        #initialization of Izihikevich parameters
        re = np.array(np.random.rand(self.n_neurons), dtype=np.double) # uniformly distributed random doubles
        voltage = -65.0 * np.ones(self.n_neurons, dtype=np.double)
        voltage_reset = -65+5*(re**2)
        recov_reset = 8-6*(re**2)
        recov_scale = 0.02+0.001*re
        recov_sensitivity = 0.2+0.001*re
        recov = recov_sensitivity * voltage  
        connect_matrix = self.connectivity.build(self.network_shape[2], self.network_shape[1], angle_connect_strength, spatial_connect_strength, total_connect_strength)
        
        #creating an instance of the Izhikevich neural model
        self.izhikevich = Izhikevich(voltage, recov, voltage_reset, recov_reset, recov_scale, recov_sensitivity, decay, thresh, connect_matrix)
        
    def _preprocess_input(self, stimulus, filter_step, filter_start_width, filter_start_height):
        '''
        Detect angles on the stimulus picture and return flattened output.
        '''
        filtered = self.connectivity.detect_angles(stimulus, filter_step, filter_start_width, filter_start_height)
        filtered[filtered < 2] = 0
        return filtered / 2
    
    def run(self, length):
        voltage, recovery, firings = self.izhikevich.simulate(self.neural_input, length)
        return voltage, recovery, firings
    
    def angle_populations(self):
        '''
        Divide all neurons into subpopulations, so that neurons which detected the same angle are in the same population.
        Output:
            indices of neurons in each population.
        '''
        n_angles = self.n_neurons // self.network_shape[0]
        nonzero = np.nonzero(self.neural_input)[0]
        populations = []
        
        for i in range(self.network_shape[0]):
            pop = nonzero[(nonzero >= n_angles * i) & (nonzero < n_angles * (i+1))]
            populations.append(list(pop))
        return populations
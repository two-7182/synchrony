#!/usr/bin/env python 3.9.7
# -*- coding: utf-8 -*-

"""Simulation module for the research project 'Cortical Spike Synchrony as 
a Measure of Contour Uniformity', as part of the RTG computational cognition, 
Osnabrueck University, Germany."""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'viktoriia.zemliak@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

#==============================================================================#
#========================|          Imports           |========================#
#==============================================================================#

#standard library imports
import os 

#related third party imports
import matplotlib.pyplot as plt 
import numpy as np
import yaml

#local imports
from network import Connectivity
from model import Izhikevich
from image_preprocessor import ImagePreprocessor
from draw import line_45_joint

#==============================================================================#
#=======================|          Simulation          |=======================#
#==============================================================================#

class Simulation:
    def __init__(self, stimulus, filters, 
                 angle_connect_strength=0.5, spatial_connect_strength=0.5, total_connect_strength=0.5,
                 inh=0.2, inh_weight=2.0, 
                 random_noise=0.0, input_firing_prob=1.0, random_seed=None,
                 input_strength=1.0, ini_standard=False
                 ):
        """Builds the connections network and runs the simulation of a given length.
        
        Args:
            stimulus = the input stimulus image
            filters = convolution filters for angles recognition
            angle_connect_strength = the strength of angular connections between neurons
            spatial_connect_strength = the strength of spatial connections between neurons
            total_connect_strength = the strength of final connections between neurons
            inh = proportion of inhibitory neurons
            inh_weight = scaling factor for the inhibitory connections
            random_noise = amount of random noie in the model
            input_firing_prob = probability of firing of the input neurons
            random_seed = random seed
        """

        #initialize the Connectivity class with filters
        self.connectivity = Connectivity(filters)
        
        #input preprocessing
        self.neural_input = stimulus
        
        #initialize helper simulation parameters
        network_shape = self.neural_input.shape
        n_neurons_exc = np.prod(self.neural_input.shape)
        n_neurons_inh = int(n_neurons_exc*inh)  
        
        #reshape neural input array for convenience
        self.neural_input = self.neural_input.reshape(-1)
        
        #add inhibitory neurons to total input
        if n_neurons_inh > 0:
            inh_input = np.ones((n_neurons_inh,)) * self.neural_input[self.neural_input>0].min()
            self.neural_input = np.concatenate((self.neural_input, inh_input))
        
        #initialize connectivity matrix
        connect_matrix = self.connectivity.build(width=network_shape[2], height=network_shape[1], 
                                                 angle_connect_strength=angle_connect_strength, 
                                                 spatial_connect_strength=spatial_connect_strength, 
                                                 total_connect_strength=total_connect_strength,
                                                 n_neurons_exc=n_neurons_exc, n_neurons_inh=n_neurons_inh, inh_weight=inh_weight
                                                 )
        
        #create an instance of the Izhikevich neural model
        self.izhikevich = Izhikevich(connect_matrix=connect_matrix, 
                                     n_neurons_exc = n_neurons_exc, n_neurons_inh = n_neurons_inh,
                                     ini_standard=ini_standard, input_strength=input_strength,
                                     random_noise=random_noise, input_firing_prob=input_firing_prob, random_seed=random_seed
                                     )
    
    #======================================================================#
    #======================|     Run Simulation     |======================#
    #=======================================================================#

    def run(self, length=100, verbose=False):
        voltage, recovery, firings = self.izhikevich.simulate(self.neural_input, length, verbose)
        return voltage, recovery, firings
        
#==============================================================================#
#==========================|          Main          |==========================#
#==============================================================================#
    
if __name__ == '__main__':

    #========================|     Parameter      |========================#
    #import experiment configurations
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #preprocessor parameter
    angular_resolution =  config["preprocessing"]["angle_resolution"]
    filter_size = config["preprocessing"]["filter_size"]
    image_path = config["preprocessing"]["input_image"]
    image_array = line_45_joint(width=28, height=28, strength=255, length=3)

    #simulation parameters
    inh = 0.2
    length = 1000
    ini_standard = True
    
    #======================|     Initialization     |======================#
    #preprocessor instantiation and preprocessing of image
    preprocessor = ImagePreprocessor(angular_resolution, filter_size)
    stimulus = preprocessor.preprocess(image_array, plot_substeps=False)

    #initialize simulation object
    print('Initialize the simulation')
    sim = Simulation(stimulus=stimulus, filters=preprocessor.angle_filters, inh=inh, ini_standard=ini_standard)

    #======================|     Run Simulation     |======================#
    print('Simulation started...')
    voltage, recovery, firings = sim.run(length, True)
    
    #=========================|     Plotting     |=========================#
    #plot voltage
    h, w = voltage.shape
    fig, ax = plt.subplots(figsize=(8,8))
    plt.imshow(voltage) 
    plt.colorbar()
    ax.set_aspect(w/h)
    
    #save the plot
    filename = 'voltage.png'
    fig.savefig(filename)
    print(f'Plot {filename} saved')
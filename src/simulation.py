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

class SimulationExperiment:

    def __init__(self, config_file_path):
        """Builds the connections network and runs the simulation of a given length.
         
        Args:
            config_file_path = file path to the yaml config file
        """

        #========================|     Parameter      |========================#
        #import experiment configurations
        abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file_path)
        with open(abs_file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)


        #======================|     Initialization     |======================#
        #preprocessor instantiation and preprocessing of image
        preprocessor = ImagePreprocessor(
            config["preprocessing"]["angle_resolution"], 
            config["preprocessing"]["filter_size"])

        #initialize the Connectivity class with filters
        connectivity = Connectivity(preprocessor.angle_filters)
        
        #input preprocessing (replace array with config["preprocessing"]["input_image"])
        image_array = line_45_joint(width=28, height=28, strength=255, length=3)
        self.neural_input = preprocessor.preprocess(image_array, plot_substeps=False)
        
        #initialize helper simulation parameters
        network_shape = self.neural_input.shape
        n_neurons_exc = np.prod(self.neural_input.shape)
        n_neurons_inh = int(n_neurons_exc * config["connectivity"]["inh"])  
        
        #reshape neural input array for convenience
        self.neural_input = self.neural_input.reshape(-1)
        
        #add inhibitory neurons to total input
        if n_neurons_inh > 0:
            inh_input = np.ones((n_neurons_inh,)) * self.neural_input[self.neural_input>0].min()
            self.neural_input = np.concatenate((self.neural_input, inh_input))
        
        #initialize connectivity matrix
        connect_matrix = connectivity.build(width=network_shape[2], height=network_shape[1], 
                                                 angle_connect_strength=config["connectivity"]["angle_connect_strength"], 
                                                 spatial_connect_strength=config["connectivity"]["spatial_connect_strength"], 
                                                 total_connect_strength=config["connectivity"]["total_connect_strength"],
                                                 n_neurons_exc=n_neurons_exc, n_neurons_inh=n_neurons_inh, 
                                                 inh_weight=config["connectivity"]["inh_weight"])
        
        #create an instance of the Izhikevich neural model
        self.izhikevich = Izhikevich(connect_matrix=connect_matrix, 
                                     n_neurons_exc = n_neurons_exc, 
                                     n_neurons_inh = n_neurons_inh,
                                     ini_standard=config["model"]["ini_standard"], 
                                     input_strength=config["model"]["input_strength"],
                                     random_noise=config["model"]["random_noise"], 
                                     input_firing_prob=config["model"]["input_firing_prob"], 
                                     random_seed=config["simulation"]["random_seed"])
    
    #======================================================================#
    #======================|     Run Simulation     |======================#
    #=======================================================================#

    def run(self, length=1000, verbose=False):
        voltage, recovery, firings = self.izhikevich.simulate(self.neural_input, length, verbose)
        return voltage, recovery, firings
        
#==============================================================================#
#==========================|          Main          |==========================#
#==============================================================================#
    
if __name__ == '__main__':

    #======================|     Run Simulation     |======================#
    print('Initialize the simulation')
    simulation = SimulationExperiment('config.yaml')

    print('Simulation started...')
    voltage, recovery, firings = simulation.run(length=10, verbose=True)
    
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
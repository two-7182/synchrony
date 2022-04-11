# -*- coding: utf-8 -*-

"""
Connectivity class for the research project 'Cortical Spike Synchrony as 
a Measure of Contour Uniformity', as part of the RTG computational cognition, 
Osnabrueck University, Germany.
"""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'viktoriia.zemliak@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

import itertools
import matplotlib.pyplot as plt 
import numpy as np
from scipy import signal, spatial
import seaborn as sns

class Connectivity:
    def __init__(self, filters):
        '''
        Class for building a matrix of horizontal connections between neurons.
        Args:
            filters = convolution filters for angles detection
        '''
        self.filters = filters
    
    def spatial_connect(self, width, height):
        '''
        Calculating spatial connectivity between all neurons.
        Args:
            width = width of the grid with neurons
            height = height of the grid with neurons
        '''
        #create an array with all neuron coordinates
        neurons_coord = np.array(list(itertools.product(range(height), range(width))))
        
        #calculate Chebyshev distance between each pair
        distances = spatial.distance.cdist(neurons_coord, neurons_coord, 'chebyshev')
        
        #duplicate the results for each group of neurons recognizing a specific angle
        distances = np.tile(distances, (len(self.filters),len(self.filters)))
        
        distances[distances < 0.5] = 0
        
        #return inverted distance: the bigger distances, the weaker connections
        return 1/(distances+1)
    
    def angle_connect(self, width, height):
        '''
        Calculating angular connectivity between all neurons. 
        Bigger angle difference -> weaker connection.
        Args:
            width = width of the grid with neurons
            height = height of the grid with neurons
        '''
        def angle_diff(x,y):
            '''
            Calculate difference between two angles up to 180 degrees
            Args:
                x = one angle of interest
                y = another angle of interest
            '''
            x,y = sorted([x,y])
            abs_diff = 180 - abs(abs(x-y) - 180)
            #if (x == 0 and y > 90) or (x < 90 and y == 180):
            #    return (180 - abs_diff)/45
            return abs_diff   
        
        #get angle resolution
        angle_res = 180//len(self.filters)
        
        #list of angles detected, e.g. [0, 45, 90, 135]
        angles = [angle_res*i for i in range(len(self.filters))]
        
        #number of neurons processing each angle
        vec_len = width * height
        
        #create an empty array for angle differences between each pair of neurons
        angle_diffs = np.zeros((vec_len*len(self.filters), vec_len*len(self.filters)))
        
        #calculate angle difference between each pair of neurons
        for i in range(len(self.filters)):
            for j in range(len(self.filters)):
                angle_diffs[vec_len*i:vec_len*(i+1),vec_len*j:vec_len*(j+1)] = angle_diff(angles[i], angles[j]) / angle_res
        
        #return inverted differences: the bigger differences, the weaker connections      
        return 1/(angle_diffs+1)
    
    def build(self, width, height, angle_connect_strength=0.5, spatial_connect_strength=0.5, total_connect_strength=0.5):
        '''
        Build a connection matrix which takes into account both spatial distance and angular difference between all pairs of neurons.
        Args:
            width = width of the grid with neurons
            height = height of the grid with neurons
            angle_connect_strength = weights of the angle connection values
            spatial_connect_strength = weights of the spatial connection values
            total_connect_strength = scaling variable for the resulting connection values
        '''
        #count spatial connection weights
        self.spatial_connect_matrix = self.spatial_connect(width, height) 
        
        #count angle connection weights
        self.angle_connect_matrix = self.angle_connect(width, height)
        
        #count resulting connection weights
        connect = (self.spatial_connect_matrix*spatial_connect_strength + self.angle_connect_matrix*angle_connect_strength) * total_connect_strength
        #connect = self.spatial_connect_matrix * self.angle_connect_matrix * total_connect_strength
        np.fill_diagonal(connect, 0) #turn connections of neurons to themselves to zero
        return connect
    
    def detect_angles(self, img, step=1, width_start=0, height_start=0):
        '''
        Detect angles of the input image wih use of the convolutional filters.
        Args:
            img = input image
            step = convolution step
            width_start = x coordinate of the starting point of the convolution filter
            height_start = y coordinate of the starting point of the convolution filter
        '''
        filtered = np.zeros((len(self.filters), img.shape[0], img.shape[1]))

        for i, f in enumerate(self.filters):
            filtered[i] = signal.convolve(img, f, mode='same', method='direct')
        
        return filtered[:, height_start::step, height_start::step]
        
if __name__ == "__main__": 

    #connectivity parameters
    filters = [[[0,0],[1,1]], [[1,0],[0,1]], [[0,1],[0,1]], [[0,1],[1,0]]]
    width, height = 5, 5
    angle_connect_strength = 0.5
    spatial_connect_strength = 0.5
    total_connect_strength = 0.5
    
    #itinialize Connectivity class
    connectivity = Connectivity(filters)
    
    #build the connectivity matrix
    connect_matrix = connectivity.build(width=width,
                                        height=height,
                                        spatial_connect_strength=spatial_connect_strength,
                                        angle_connect_strength=angle_connect_strength,
                                        total_connect_strength=total_connect_strength)
        
    #helper functions for plotting and saving the resulting connectivity matrix
    def set_visible_labels(labels, step=5):
        '''
        Decrease the number of labels visible on the plot.
        '''
        for label in labels:
            if np.int(label.get_text()) % step == 0:  
                label.set_visible(True)
            else:
                label.set_visible(False)
                
    def plot_connectivity(connect_matrix, filename='connect_matrix.png', show=False):
        '''
        Plot the connectivity matrix.
        '''
        
        fig = plt.figure(figsize=(15,15))
        ax = sns.heatmap(connect_matrix, vmax=connect_matrix.max(), square=True,  cmap="YlGnBu")
        
        #construct the plot title from the filename
        plot_title = filename.split('.')[0].replace('_', ' ').capitalize()
        ax.set_title(plot_title)
    
        ax.invert_yaxis()
        
        #create custom ticks
        ticks = [list(range(0,width*height)) for i in range(len(filters))]
        ticks = sum(ticks, [])
        
        #set x ticks
        ax.set_xticks(range(100))
        ax.set_xticklabels(ticks)
            
        #set y ticks
        ax.set_yticks(range(100))
        ax.set_yticklabels(ticks)            
                
        set_visible_labels(ax.get_xticklabels())
        set_visible_labels(ax.get_yticklabels())

        if show:
            plt.show()
        fig.savefig(filename)
        print(f'Plot {filename} saved')
        
    #plot and save connectivity matrices
    plot_connectivity(connect_matrix, 'total_connect_matrix.png', True)
    plot_connectivity(connectivity.spatial_connect_matrix, 'spatial_connect_matrix.png', False)
    plot_connectivity(connectivity.angle_connect_matrix, 'angle_connect_matrix.png', False)
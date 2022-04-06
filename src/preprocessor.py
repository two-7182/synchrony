#!/usr/bin/env python 3.9.7
# -*- coding: utf-8 -*-

"""
Preprocessing class for the research project 'Cortical Spike Synchrony as a 
Measure of Contour Uniformity', as part of the RTG computational cognition, 
Osnabrueck University, Germany.
"""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'julius.mayer@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

#==============================================================================#
#========================|          Imports           |========================#
#==============================================================================#

#standard library imports
import os 
import warnings

#related third party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter

#==============================================================================#
#======================|          Preprocessor          |======================#
#==============================================================================#

class Preprocessor:

    def __init__(self, angular_resolution, filter_size, set_filter_size_to_min=False):
        """Preprocessing class for the research project 'Cortical Spike Synchrony as a 
        Measure of Contour Uniformity'
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection
            set_filter_size_to_min = whether to use the minimal viable filter size instead
        """

        #list of filters for angle detection
        self.angle_filters = self._get_angle_filters(angular_resolution, filter_size, set_filter_size_to_min)
        self.angle_labels = self._get_angle_labels(angular_resolution)
        

    #======================================================================#
    #========================|     Functions      |========================#
    #======================================================================#

    def preprocess(self, image_path, plot_substeps=True):
        """Preprocess the image file specified by the image path, first from 
        RGB to grayscale. Then apply Sobel filter on the intensity values to
        gain edge detection. Threshold result for binary image and convolve
        angular filters for angle detection maps.
        
        args:
            image_path = file path of the image input
            plot_substeps = whether to compile the substeps, plot and save them
        """
        
        with Image.open(image_path) as image_input:
            print('Start preprocessing for ' + os.path.split(image_path)[1] +f' with size {image_input.width} x {image_input.height}')

            image_gray   = image_input.convert("L")
            image_edges  = image_gray.filter(ImageFilter.FIND_EDGES)
            image_binary = image_edges.point(lambda pixel_value : 255 if pixel_value > threshold_otsu(np.array(image_gray)) else 0, mode = "1")
            image_angles = [convolve(np.array(image_binary), angle_filter, mode='same', method='direct') for i, angle_filter in enumerate(self.angle_filters)]

            if plot_substeps: 
                plot_label = ["image preprocessing", "original image", "intensity values", "edge detection", "binary image"]
                file_path = os.path.split(image_path)[0]+f"/preprocessing_steps_{os.path.split(image_path)[1]}"
                self._plot_images(file_path, [image_input, image_gray, image_edges, image_binary], plot_label)

                plot_label =  self.angle_labels
                file_path = os.path.split(image_path)[0]+f"/stimulus_mapping_{os.path.split(image_path)[1]}"
                self._plot_images(file_path, image_angles, plot_label)
            
        return image_angles

    def _get_angle_filters(self, angular_resolution, filter_size, set_filter_size_to_min):
        """Create filters to detect angles in the image
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection
            set_filter_size_to_min = whether to use the minimal viable filter size instead
        """

        #Exception handling for angular resolution
        if not 1 < angular_resolution < 90:
            raise ValueError('Angular resolution is out of range, must be between 1 and 90.')

        elif 180 % angular_resolution != 0:
            warnings.warn("Angular resolution doesn't devide evenly")

        #Exception handling for filter size
        if set_filter_size_to_min:
            filter_size = self._get_min_filter_size(angular_resolution, filter_size)

        elif filter_size < self._get_min_filter_size(angular_resolution, filter_size):
            raise ValueError('Filter size is too small for specified angular resolution')


        #TODO: write code here (add dynamic filter creation here)
        # edge detection filters for 0, 45, 90, 135 degree 
        # (replace with dynamic generation)
        angle_filters = [[[0,0],[1,1]], [[1,0],[0,1]], [[0,1],[0,1]], [[0,1],[1,0]]]

        return angle_filters  

    def _get_min_filter_size(self, angular_resolution, filter_size):
        """Calculate minimal filter size given by the specified angular resolution
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection
        """

        #TODO: write code to calculate min filter size
        min_filter_size = 0
        return min_filter_size

    def _get_angle_labels(self, angular_resolution):

        #TODO: create dynamical list of angle labels
        return ["angle detection maps", "0 degree", "45 degree", "90 degree", "135 degree"]


    def _plot_images(self, file_path, images, plot_label):
        """Compile plot and save all substeps into one figure
        
        args:
            file_path = file path of the image to be saved under
            images = list of images to be plotted and saved
            descriptions = list containing figure title and description of the images 
        """

        fig, ax = plt.subplots(nrows=2, ncols=2)
        for i, axi in enumerate(ax.flat):
            axi.axis("off")
            axi.set_title(plot_label[i+1])
            axi.imshow(images[i])
        plt.suptitle(plot_label[0])
        plt.tight_layout()
        plt.savefig(file_path)
        plt.show()

#==============================================================================#
#==========================|          Main          |==========================#
#==============================================================================#

if __name__ == "__main__": 

    #preprocessor parameter
    angular_resolution = 10
    filter_size = 100
    set_filter_size_to_min = False
    image_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data/laptop2.png")

    #preprocessor instantiation
    preprocessor = Preprocessor(angular_resolution, filter_size, set_filter_size_to_min)

    #image preprocessing
    result = preprocessor.preprocess(image_path)
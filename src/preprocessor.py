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
from typing import List, overload
from functools import singledispatch, update_wrapper


#related third party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter

#local imports
from draw import line_45_joint

#==============================================================================#
#=========================|          Utils           |=========================#
#==============================================================================#

def instance_method_singledispatch(func):
    """Small wrapper function to allow for singledispatch of instance methods"""
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)
    return wrapper

def plot_images(images, plot_label):
    """Compile plot and save all substeps into one figure
    
    args:
        images = list of images to be plotted and saved
        plot_label = list containing figure title and description of the images 
    """

    file_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), f"data/{plot_label[0]}")

    #TODO: create dynamic size plotting for more angle filters
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
#======================|          Preprocessor          |======================#
#==============================================================================#

class Preprocessor:

    @overload
    def __init__(self, angular_resolution:int, filter_size:int) -> None:
        ...
    @overload
    def __init__(self, angular_resolution:int, filter_size:None) -> None:
        ...
    def __init__(self, angular_resolution, filter_size):
        """Preprocessing class for the research project 'Cortical Spike 
        Synchrony as a Measure of Contour Uniformity'
        
        args:
            angular_resolution = angular resolution of the filters for 
                                 angle detection
            filter_size = size of the filters for angle detection, if 
                                 None minimal viable filter size instead
        """

        #lists of filters and labels for angle detection
        self.angle_filters = self._get_angle_filters(filter_size, angular_resolution)
        self.angle_labels = self._get_angle_labels(angular_resolution)
        
    #======================================================================#
    #======================|     Preprocessing      |======================#
    #======================================================================#
    
    @instance_method_singledispatch
    def preprocess(self, image_input:Image.Image, plot_substeps:bool=True) -> np.ndarray:
        """Preprocess the image, first from RGB to grayscale. Then apply a Sobel 
        filter on the intensity values to gain edge detection. Threshold result 
        for binary image and convolve angular filters for angle detection maps.
        
        args:
            image_input = image input as a Pillow image object
            plot_substeps = whether to compile the substeps, plot and save them

        Possibly enhance image processing with:
        imageObject.filter(ImageFilter.EDGE_ENHANCE) 
        imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
            (all edge image angle maps are more prominent)
        imageObject.filter(ImageFilter.SMOOTH)
            (all edge image angle maps are less prominent)
        => effects seem unhelpfull so far
        """

        #preprocessing pipeline for input image
        image_gray   = image_input.convert("L")
        image_edges  = image_gray.filter(ImageFilter.FIND_EDGES)
        threshold    = threshold_otsu(np.array(image_edges))
        threshold_fn = lambda pixel_value : 1 if pixel_value > threshold else 0
        image_binary = image_edges.point(threshold_fn, mode = "1")
        image_angles = np.array([convolve(image_binary, angle_filter, mode='same', 
            method='direct') for i, angle_filter in enumerate(self.angle_filters)])
        image_angles[np.array(image_angles) < 2] = 0

        #plot and save substeps if requested
        if plot_substeps:
            plot_label = ["image_preprocessing", "original image", 
                "intensity values", "edge detection", "binary image"]
            plot_images([image_input, image_gray, image_edges, image_binary], plot_label)
            plot_images(image_angles.tolist(), self.angle_labels)
            
        return image_angles

    @preprocess.register(str)
    def _(self, image_input:str, plot_substeps:bool=True) -> np.ndarray:
        """Wrapper for overloading the preprocess method
        
        args:
            image_input = file path of the image input
            plot_substeps = whether to compile the substeps, plot and save them
        """

        try:
            return self.preprocess(Image.open(image_input), plot_substeps)

        except FileNotFoundError:
            print('File does not exist')

    @preprocess.register(np.ndarray)
    def _(self, image_input:np.ndarray, plot_substeps:bool=True) -> np.ndarray:
        """Wrapper for overloading the preprocess method
        
        args:
            image_input = image input as a numpy array
            plot_substeps = whether to compile the substeps, plot and save them
        """

        return self.preprocess(Image.fromarray(image_input), plot_substeps)

    #======================================================================#
    #=================|     Angle Filter Generation      |=================#
    #======================================================================#

    @instance_method_singledispatch
    def _get_angle_filters(self, filter_size:int, angular_resolution:int) -> list[str]:
        """Create filters to detect angles in the image
        
        args:
            filter_size = size of the filters for angle detection
            angular_resolution = angular resolution of the filters for angle detection
        """

        #Exception handling for angular resolution
        if not 1 < angular_resolution < 90:
            raise ValueError('Angular resolution is out of range, must be between 1 and 90.')

        elif 180 % angular_resolution != 0:
            warnings.warn("Angular resolution doesn't devide evenly")

        #Exception handling for filter size
        elif filter_size < self._get_min_filter_size(angular_resolution):
            raise ValueError('Filter size is too small for specified angular resolution')

        else:
            #TODO: write code here (add dynamic filter creation here)
            # edge detection filters for 0, 45, 90, 135 degree 
            # (replace with dynamic generation)
            angle_filters = [[[0,0],[1,1]], [[1,0],[0,1]], [[0,1],[0,1]], [[0,1],[1,0]]]
            print(type(angle_filters))

        return angle_filters  

    @_get_angle_filters.register
    def _(self, filter_size:None, angular_resolution:int) -> list[str]:
        """"Wrapper for overloading the _get_angle_filters method
        
        args:
            filter_size = omitted (None)
            angular_resolution = angular resolution of the filters for angle detection
        """

        return self._get_angle_filters(self._get_min_filter_size(angular_resolution), angular_resolution)

    def _get_min_filter_size(self, angular_resolution:int) -> int:
        """Calculate minimal filter size given by the specified angular resolution
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection
        """

        #TODO: write code to calculate min filter size
        min_filter_size = 25
        return min_filter_size

    def _get_angle_labels(self, angular_resolution:int) -> list[str]:
        """Create a dynamical list of the angle labels
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
        """

        #TODO: create dynamical list of angle labels
        return ["angle_detection_maps", "0 degree", "45 degree", "90 degree", "135 degree"]

#==============================================================================#
#==========================|          Main          |==========================#
#==============================================================================#

if __name__ == "__main__": 

    #preprocessor parameter
    angular_resolution = 10
    filter_size = 100 #can also be 'None' for min. viable filter size

    #different possible types of input instances
    image_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data/laptop2.png")
    image_object = Image.open(image_path)
    image_array = line_45_joint(width=28, height=28, strength=255, length=3)

    #preprocessor instantiation
    preprocessor = Preprocessor(angular_resolution, filter_size)

    #access angle filters for connectivity matrix
    angle_filters = preprocessor.angle_filters

    #preprocessing can take any of image_path, image_object or image_array
    result = preprocessor.preprocess(image_path)
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
from math import radians, cos, sin
from skimage.draw import line_nd

#local imports
from draw import line_45_joint

#==============================================================================#
#=========================|          Utils           |=========================#
#==============================================================================#

def singledispatch_instance_method(func):
    """Small wrapper function to allow for singledispatch of instance methods"""

    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)
    return wrapper

def plot_images(images, plot_label):
    """Compile plot from list of images and save all substeps into one figure

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

    @singledispatch_instance_method
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

    @singledispatch_instance_method
    def _get_angle_filters(self, filter_size:int, angular_resolution:int) -> list[str]:
        """Create filters to detect angles in the image

        args:
            filter_size = size of the filters for angle detection
            angular_resolution = angular resolution of the filters for angle detection
        """

        def get_endpoints(angle, size):
            """ Compute endpoints of the vector via a rotation matrix

            args:
                angle = the angle for which the endpoints should be computed
                size = size of the filters for angle detection.
            """


            rad_angle = radians(angle)
            rotation_matrix = np.array([[cos(rad_angle), sin(rad_angle)],
                                        [sin(rad_angle), cos(rad_angle)]])

            vector = np.array([size,0])
            #output of the dot product: vector
            result = rotation_matrix.dot(vector)

            # # scale the resulting vector so to reach desired size
            #(otherwise, it is too short)
            # depending on the angle size the vector should either intersect
            #with the columns of the array or the rows
            if 1< angle < 44 or angle > 136:
                #when it should intersect with the columns (x-axis),
                #so take x-value of vector
                scale = size / result[0] #bc it should reach out to the columns
            else:
                #when vector should intersect with the rows (y-axis),
                #so take y-value of vector
                scale = size / result[1]

            x2 = result[0] * scale
            y2 = result[1] * scale


            return x2,y2

        #Exception handling for angular resolution
        if not 1 < angular_resolution < 90:
            raise ValueError('Angular resolution is out of range, must be between 1 and 90.')

        elif 180 % angular_resolution != 0:
            warnings.warn("Angular resolution doesn't devide evenly")

        #Exception handling for filter size
        elif filter_size < self._get_min_filter_size(angular_resolution):
            raise ValueError('Filter size is too small for specified angular resolution')

        else:

            angle_filters = []

            # calculate the list of angles from the given angular resolution
            count = angular_resolution
            degree_list = []

            while angular_resolution <= 180:
                degree_list.append(angular_resolution)
                angular_resolution += count

            for angle in degree_list:

                # Exception handling for angles sizes and types
                if angle <= 0 or angle > 180 or isinstance(angle, int) == False:
                    raise ValueError("Degree for the filter cannot be smaller 0° or greater than 180° and must be an integer")

                elif  0 < angle <= 44:
                    x1 = filter_size -1
                    y1 = 0
                    x2, y2 = get_endpoints(angle, filter_size)
                    # convert to array
                    x2 = filter_size -1 #columns. line intersects always at last column
                    y2 = (filter_size - y2) - 1
                    if y2 < 0:
                        raise ValueError("Chosen filter size is too small for this angle")

                elif 45 <= angle <= 89:
                    x1 = filter_size -1
                    y1 = 0
                    x2, y2 = get_endpoints(angle, filter_size)
                    #convert to array
                    y2 = 0 # rows
                    x2 = x2 - 1
                    if x2 < 0:
                        raise ValueError("Chosen filter size is too small for this angle")

                elif 90 <= angle <= 134:
                    x1 = filter_size - 1
                    y1 = filter_size -1
                    x2, y2 = get_endpoints(angle, filter_size)
                    y2 = 0
                    x2 = (filter_size - abs(x2)) - 1 # take absolute value here to change from negative to positive -> no neg. values in arrays
                    if x2 <= 0:
                        raise ValueError("Chosen filter size is too small for this angle")

                elif angle == 135:
                    x1 = filter_size - 1
                    y1 = filter_size - 1
                    x2 = 0
                    y2 = 0

                #135 <= angle <= 180:
                else:
                    x1 = filter_size - 1
                    y1 = filter_size -1
                    x2, y2 = get_endpoints(angle, filter_size)
                    x2 = 0 # columns do not change- > intersect always at column 0
                    y2 = (filter_size - abs(y2)) - 1
                    if y2 <= 0:
                        raise ValueError("Chosen filter size is too small for this angle")


                conv_filter = np.zeros(shape=(filter_size,filter_size))
                angle_line = line_nd((x1,y1), (y2, x2), endpoint=True) #using line_nd in order to be able use floats
                conv_filter[angle_line] = 1
                angle_filters.append(conv_filter)

            return angle_filters


    @_get_angle_filters.register
    def _(self, filter_size:None, angular_resolution:int) -> list[str]:
        """"Wrapper for overloading the _get_angle_filters method

        args:
            filter_size = None
            angular_resolution = angular resolution of the filters for angle detection
        """

        return self._get_angle_filters(self._get_min_filter_size(angular_resolution), angular_resolution)

    def _get_min_filter_size(self, angular_resolution:int) -> int:
        """Calculate minimal filter size given by the specified angular resolution

        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection
        """

        count = angular_resolution
        filter_size = 15 #start out with default filter size
        degree_list = []
        filter_size_list = []


        while angular_resolution <= 180:
            degree_list.append(angular_resolution)
            angular_resolution += count

        for arc in degree_list:

            x, y = get_endpoints(arc, filter_size)
            x = abs(x)
            y = abs(y)

            if 0 < arc <= 44 or arc >= 135: #first and fourth case
                while y >= 0:
                    filter_size -= 1
                    x, y = get_endpoints(arc, filter_size)
                    x = round(abs(x),3)
                    y = round(abs(y),3)
                    if ((filter_size - y) - 1) <= 3:
                        filter_size_list.append(filter_size)
                        break


            elif 45 <= arc <= 89: # second case
                while x >= 0:
                    filter_size -= 1
                    x, y = get_endpoints(arc, filter_size)
                    x = round(abs(x),3)
                    y = round(abs(y),3)
                    if (x-1) <= 3:
                        filter_size_list.append(filter_size)
                        break

            else:# 90 <= arc <= 134 so third case
                while x >= 0:
                    filter_size -= 1
                    x, y = get_endpoints(arc, filter_size)
                    x = round(abs(x),3)
                    y = round(abs(y),3)
                    if ((filter_size - x)-1) <= 3:
                        filter_size_list.append(filter_size)
                        break


        min_filter_size = max(filter_size_list)
        return min_filter_size




    def _get_angle_labels(self, angular_resolution:int) -> list[str]:
        """Create a dynamical list of the angle labels

        args:
            angular_resolution = angular resolution of the filters for angle detection
        """

        count = angular_resolution
        degree_list = ["angle_detection_maps"]

        # make a list of all angles from angular_resolution and put them into a string
        while angular_resolution <= 180:
            degree_in_string = "{} {}".format(str(angular_resolution), "degree")
            angular_resolution += count
            degree_list.append(degree_in_string)

        return degree_list

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

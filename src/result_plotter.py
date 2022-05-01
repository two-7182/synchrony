#!/usr/bin/env python 3.9.7
# -*- coding: utf-8 -*-

"""Result plotter module for the research project 'Cortical Spike Synchrony 
as a Measure of Contour Uniformity', as part of the RTG computational 
cognition, Osnabrueck University, Germany."""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'viktoriia.zemliak@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

#==============================================================================#
#========================|          Imports           |========================#
#==============================================================================#

#standard library imports
from ctypes.wintypes import PLONG
import os 
import warnings
from typing import overload
from functools import singledispatch, update_wrapper

#related third party imports
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

#==============================================================================#
#=========================|          Utils           |=========================#
#==============================================================================#

class Observable:

    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify(self, *args, **kwargs):
        for obs in self._observers:
            obs.update(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self._observers.remove(observer)

class Observer:

    def __init__(self, observables):
        for observable in observables: observable.subscribe(self)

    def update(self, observable, *args, **kwargs):
        raise NotImplementedError

#==============================================================================#
#=====================|          Resultplotter           |=====================#
#==============================================================================#

class ResultPlotter(Observer):

    def __init__(self, config, observables):
        """Builds the connections network and runs the simulation of a given length.
         
        Args:
            config = yaml config
            observables = list of observable object to subscribe to
        """
        
        super().__init__(observables)
        self.config = config

    def update(self, observable, *args, **kwargs):

        if observable is "ImagePreprocessor":
            if self.config["plotting"]["plot_substeps"]:
                self._plot_images(*args, **kwargs)
        else:
            if self.config["plotting"]["plot_substeps"]:
                self._plot_images(*args, **kwargs)


    #======================================================================#
    #======================|     Preprocessing      |======================#
    #======================================================================#

    @overload
    def _plot_images(self, images:list[Image.Image], image_or_map_labels:bool) -> None:
        ...
    def _plot_images(self, images:list[list], image_or_map_labels:bool) -> None:
        """Compile plot from list of images and save all substeps into one figure
        
        args:
            images = list of images to be plotted and saved
            image_or_map_labels = true for returning image labels, 
                                  false for returning map labels
        """

        #image labels or Create a dynamical list of the angle labels
        image_labels = ["image preprocessing steps", "original image", 
                "intensity values", "edge detection", "binary image"]
        map_labels = ["angle detection maps"] + [f"{i}°" for i in  
                np.arange(0, 180, self.config["preprocessing"]["angle_resolution"])]
        labels = image_labels if image_or_map_labels else map_labels

        file_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), f"data/{labels[0]}")

        #plot images on multiple axis
        fig, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=int(len(images)/2))
        for i, axi in enumerate(ax.flat):
            axi.axis("off")
            axi.set_title(labels[i+1])
            axi.imshow(images[i])
        plt.suptitle(labels[0])
        plt.tight_layout()
        plt.savefig(file_path)
        plt.show()

    #======================================================================#
    #======================|     Run Simulation     |======================#
    #======================================================================#

    def _plot_results(self, voltage):
        #plot voltage
        h, w = voltage.shape
        fig, ax = plt.subplots(figsize=(8,8))
        plt.imshow(voltage) 
        plt.colorbar()
        ax.set_aspect(w/h)
        
        #save the plot
        filename = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data/voltage.png')
        fig.savefig(filename)
        print(f'Plot {filename} saved')
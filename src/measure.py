# -*- coding: utf-8 -*-

"""
Meeasure class for the research project 'Cortical Spike Synchrony as 
a Measure of Contour Uniformity', as part of the RTG computational cognition, 
Osnabrueck University, Germany.
"""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'viktoriia.zemliak@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

import itertools
import numpy as np
from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from elephant.spike_train_synchrony import spike_contrast
from neo.core import SpikeTrain

class Measure:
    def __init__(self, firings, metric='van_rossum'):
        '''
        Function for calculating dissimilarity between multiple spike trains.
        Args:
            metric = metric name. Available metrics: van_rossum, victor_purpura.
            firings = list of sequences of the neuron firings.
        '''
        metrics_available = ('van_rossum', 'victor_purpura', 'spike_contrast', 'rsync')
        if metric not in metrics_available:
            raise Exception('Please select from the available metrics: van_rossum, victor_purpura, spike_contrast, rsync')
        self.metric = metric
        
        if len(firings) < 2:
            raise Exception('Please select 2 or more spike trains to compare')
        if len(set([len(f) for f in firings])) > 1:
            raise Exception('Please select spike trains of the similar length')
            
        self.firings = firings
        #print(len(self.firings), 'spike trains to compare')
        self.length = len(firings[0])
        
    def _transform_firing(self, spike_train):
        return SpikeTrain(list(np.nonzero(spike_train))[0], units='ms', t_stop=self.length)
        
    def _pairwise_distance(self, firing1, firing2):
        train1 = self._transform_firing(firing1)
        train2 = self._transform_firing(firing2)

        if self.metric == 'van_rossum':
            return van_rossum_distance((train1, train2))[0,1]
        return victor_purpura_distance((train1, train2))[0,1]
    
    def dissimilarity(self):
        '''
        Measure the distance between arbitrary amount of neurons.
        '''
        if self.metric == 'spike_contrast':
            trains = [self._transform_firing(firing) for firing in self.firings]
            return 1 - spike_contrast(trains)
        
        elif self.metric == 'rsync':
            if isinstance(self.firings, list):
                firings = np.zeros((len(self.firings), len(self.firings[0])))
                for i,f in enumerate(self.firings):
                    firings[i] = f
                self.firings = firings
                
            meanfield = np.mean(self.firings, axis=0) # spatial mean across cells, at each time
            variances = np.var(self.firings, axis=1)  # variance over time of each cell
            return 1 - np.var(meanfield) / np.mean(variances)
        
        else:
            pairs = list(itertools.combinations(range(len(self.firings)), 2))
            distances = [self._pairwise_distance(self.firings[pair[0]], self.firings[pair[1]]) for pair in pairs]
        return {'median': np.median(distances), 'mean': np.mean(distances), 'max': np.max(distances), 'min': np.min(distances)}
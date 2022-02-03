import itertools
import numpy as np
from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from neo.core import SpikeTrain

class Measure:
    def __init__(self, firings, metric='van_rossum'):
        '''
        Function for calculating dissimilarity between multiple spike trains.
        Args:
            metric = metric name. Available metrics: van_rossum, victor_purpura.
            firings = list of sequences of the neuron firings.
        '''
        metrics_available = ('van_rossum', 'victor_purpura')
        if metric not in metrics_available:
            raise Exception('Please select from the available metrics: van_rossum, victor_purpura')
        self.metric = metric
        
        if len(firings) < 2:
            raise Exception('Please select 2 or more spike trains to compare')
        if len(set([len(f) for f in firings])) > 1:
            raise Exception('Please select spike trains of the similar length')
            
        self.firings = firings
        print(len(self.firings), 'spike trains to compare')
        self.length = len(firings[0])
        
    def _pairwise_distance(self, firing1, firing2):
        train1 = SpikeTrain(list(np.nonzero(firing1))[0], units='ms', t_stop=self.length)
        train2 = SpikeTrain(list(np.nonzero(firing2))[0], units='ms', t_stop=self.length)

        if self.metric == 'van_rossum':
            return van_rossum_distance((train1, train2))[0,1]
        return victor_purpura_distance((train1, train2))[0,1]
    
    def dissimilarity(self):
        '''
        Measure the distance between arbitrary amount of neurons.
        '''
        pairs = list(itertools.combinations(range(len(self.firings)), 2))
        distances = [self._pairwise_distance(self.firings[pair[0]], self.firings[pair[1]]) for pair in pairs]
        return {'median': np.median(distances), 'mean': np.mean(distances), 'max': np.max(distances), 'min': np.min(distances)}
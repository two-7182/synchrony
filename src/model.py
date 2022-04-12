# -*- coding: utf-8 -*-

"""
Model class for the research project 'Cortical Spike Synchrony as 
a Measure of Contour Uniformity', as part of the RTG computational cognition, 
Osnabrueck University, Germany.
"""

__author__    = 'Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__email__     = 'viktoriia.zemliak@uni-osnabrueck.de'
__date__      = '01.04.2022'
__copyright__ = '(C) 2022 Julius Mayer, Viktoria Zemliak, Flora Perizonius'
__license__   = 'MIT License'

import math
import numpy as np
import scipy as sp
import time

class Izhikevich:
    def __init__(self, connect_matrix, n_neurons_exc, n_neurons_inh,
                 ini_standard=False, input_strength=1.0,
                 random_noise=0.2, input_firing_prob=1.0, random_seed=42):
        """
        A model of Izhikevich neuron.
        Args:
            connect_matrix = matrix of interneuronal connections
            n_neurons_exc = the number of excitatory neurons
            n_neurons_inh = the number of inhibitory neurons
            ini_standard 
                - True: initialize with standard parameters from [Izhikevich, 2003]
                - False: initialize with new parameters from [Korndörfer, Ullner, García-Ojalvo, & Pipa, 2007]
            input_strength = scaling factor for the total input to the model
            random_noise = the amount of random noise
            input_firing_prob = probability of firing of the input neurons
            random_seed = random seed

        Model parameters:
            voltage = membrane potential
            recov = recovery variable
            voltage_reset = after-spike reset value of voltage
            recov_reset = after-spike reset increment of recovery
            recov_scale = time scale of recovery
            recov_sensitivity = sensitivity of recovery to subthreshold oscillations
            connect_matrix = matrix of interneuronal connections
        """
        self.connect_matrix = connect_matrix
        self.random_noise = random_noise
        self.input_firing_prob = input_firing_prob
        self.input_strength = input_strength
        
        self.decay = 0.1
        self.threshold = 30
        
        if random_seed and isinstance(random_seed, int):
            np.random.seed(random_seed)    
        
        n_neurons_total = n_neurons_inh + n_neurons_exc
        if ini_standard == True:
            #initialization of Izihikevich parameters FROM IZHIKEVICH 2003
            re = np.array(np.random.rand(n_neurons_exc), dtype=np.double) # uniformly distributed random doubles
            ri = np.array(np.random.rand(n_neurons_inh), dtype=np.double) 
            self.voltage = -65.0 * np.ones(n_neurons_total, dtype=np.double)
            self.voltage_reset = np.concatenate((-65+5*(re**2),-65+0.5*(ri**2)))
            self.recov_reset = np.concatenate((8-6*(re**2),  2+0.5*(ri**2)))
            self.recov_scale = np.concatenate((0.02+0.001*re, 0.02+0.02*ri)) 
            self.recov_sensitivity = np.concatenate((0.2+0.001*re, 0.25-0.05*ri))
            self.recov = self.recov_sensitivity * self.voltage  
            
        elif ini_standard == False:
            #initialization of Izihikevich parameters FROM KORNDÖRFER 2017
            self.re = np.array(np.random.rand(n_neurons_exc), dtype=np.double) # uniformly distributed random doubles
            self.ri = np.array(np.random.rand(n_neurons_inh), dtype=np.double)
            self.voltage = -65.0 * np.ones(n_neurons_total, dtype=np.double)
            self.voltage_reset = -65.0 * np.ones(n_neurons_total, dtype=np.double)
            self.recov_reset = 6.0 * np.ones(n_neurons_total, dtype=np.double)
            self.recov_scale = 0.02 * np.ones(n_neurons_total, dtype=np.double)
            self.recov_sensitivity = 0.2 * np.ones(n_neurons_total, dtype=np.double)
            self.recov = self.recov_sensitivity * self.voltage  
    
    def _time_step(self, voltage, recov, all_input, substeps=2):
        '''
        Worker function for simulation. given parameters and current state variables, compute next ms
        '''
        fired = voltage > self.threshold # array of indices of spikes
        voltage_next = voltage # next step of membrane potential
        recov_next = recov # next step of recovery variable
        
        ### Action potentials ###
        voltage[fired] = self.threshold
        voltage_next[fired] = self.voltage_reset[fired] # reset the voltage of any neuron that fired to c
        recov_next[fired] = recov[fired] + self.recov_reset[fired] # reset the recovery variable of any fired neuron
        # sum spontanous thalamic input and weighted inputs from all other neurons
        input_next = all_input + np.sum(self.connect_matrix[:,fired], axis=1) 
        
        ### Step forward ###
        for i in range(substeps):  # for numerical stability, execute at least two substeps per ms
            voltage_next += (1.0/substeps) * (0.04*(voltage_next**2) + (5*voltage_next) + 140 - recov + input_next)
            recov_next += (1.0/substeps) * self.recov_scale * (self.recov_sensitivity*voltage_next - recov_next)
            
        voltage_next[voltage_next > 70] = 70
        voltage_next[voltage_next < -100] = -100
        
        recov_next[recov_next > 70] = 70
        recov_next[recov_next < -100] = -100   
        
        return voltage_next, recov_next, fired
    
    def thalamic_input(self, signal):
        '''
        Generates randomized thalamic input: each nonzero neuron spikes with agiven probability.
        Args:
            signal = input signal
            prob = spiking probability
        '''
        out_signal = np.zeros_like(signal)
        idx_nonzero = np.nonzero(signal)[0]
        spikes = np.random.choice(len(idx_nonzero), int(np.floor(self.input_firing_prob * len(idx_nonzero))), replace=False)
        out_signal[idx_nonzero[spikes]] = 1
        return out_signal
        
    def simulate(self, neural_input, length, verbose=True):
        '''
        Simulates network evolution with spike-timing-dependent plasticity. 
        Args:
            neural_input = input to the model
            length = simulation length in ms
        Output:
            Nxlength matrix of membrane voltages over time
            Nxlength matrix of recovery variables over time
            Nxlength matrix of spikes over time
        '''
        n_neurons = len(neural_input)
        
        all_input = np.zeros(shape=(n_neurons,))
        voltage_out = np.zeros((n_neurons, length), dtype=np.double)
        voltage_out[:,0] = self.voltage  # initial voltages
        recov_out = np.zeros((n_neurons, length), dtype=np.double)
        recov_out[:,0] = self.recov #initial recovery
        firings_out = np.zeros((n_neurons, length), dtype=np.double)
        
        t0 = time.perf_counter() #py3.8

        for t in range(1,length):
            random_inp = (self.thalamic_input(neural_input) + np.random.rand(n_neurons) * self.random_noise) * self.input_strength
            all_input += random_inp
            all_input *= 1 - self.decay
            voltage_out[:,t], recov_out[:,t], firings_out[:,t] = self._time_step(voltage_out[:,t-1],
                                                                    recov_out[:,t-1], all_input)
            
            if verbose and t % 100 == 0:
                print(f"Simulated {str(t)} ms of braintime in {str(time.perf_counter()-t0)} s of computer time.") #py3.8
                
        t1 = time.perf_counter() 
        print(f"Simulation took {str((t1-t0))} s")
        return voltage_out, recov_out, firings_out
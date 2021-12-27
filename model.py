import numpy as np
import scipy as sp
import time
from network import Connectivity

class Izhikevich:
    def __init__(self, voltage, recov, voltage_reset, recov_reset, recov_scale, recov_sensitivity, decay, thresh, connect_matrix):
        """
        A model of Izhikevich neuron.
        Args:
            voltage = membrane potential
            recov = recovery variable
            voltage_reset = after-spike reset value of voltage
            recov_reset = after-spike reset increment of recovery
            recov_scale = time scale of recovery
            recov_sensitivity = sensitivity of recovery to subthreshold oscillations
            connect_matrix = matrix of interneuronal connections
        """
        self.voltage = voltage
        self.recov = recov
        self.voltage_reset = voltage_reset
        self.recov_reset = recov_reset
        self.recov_scale = recov_scale
        self.recov_sensitivity = recov_sensitivity
        self.connect_matrix = connect_matrix
        self.decay = decay
        self.thresh = thresh
    
    def _time_step(self, voltage, recov, all_input, substeps=2):
        '''
        Worker function for simulation. given parameters and current state variables, compute next ms
        '''
        fired = voltage > self.thresh # array of indices of spikes
        voltage_next = voltage # next step of membrane potential
        recov_next = recov # next step of recovery variable
        
        ### Action potentials ###
        voltage[fired] = self.thresh
        voltage_next[fired] = self.voltage_reset[fired] # reset the voltage of any neuron that fired to c
        recov_next[fired] = recov[fired] + self.recov_reset[fired] # reset the recovery variable of any fired neuron
        # sum spontanous thalamic input and weighted inputs from all other neurons
        input_next = all_input + np.sum(self.connect_matrix[:,fired], axis=1) 
        
        ### Step forward ###
        for i in range(substeps):  # for numerical stability, execute at least two substeps per ms
            voltage_next += (1.0/substeps) * (0.04*(voltage_next**2) + (5*voltage_next) + 140 - recov + input_next)
            recov_next += (1.0/substeps) * self.recov_scale * (self.recov_sensitivity*voltage_next - recov_next)
        return voltage_next, recov_next, fired
    
    @staticmethod
    def thalamic_input(signal, prob=0.6):
        '''
        Generates randomized thalamic input: each nonzero neuron spikes with agiven probability.
        Args:
            signal = input signal
            prob = spiking probability
        '''
        out_signal = np.zeros_like(signal)
        idx_nonzero = np.nonzero(signal)[0]
        spikes = np.random.choice(len(idx_nonzero), int(np.floor(prob * len(idx_nonzero))), replace=False)
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
        
        t0 = time.clock()

        for t in range(1,length):
            random_inp = Izhikevich.thalamic_input(neural_input) + np.random.rand(n_neurons) * 0.3
            all_input += random_inp
            all_input -= all_input * self.decay
            voltage_out[:,t], recov_out[:,t], firings_out[:,t] = self._time_step(voltage_out[:,t-1],
                                                                    recov_out[:,t-1], all_input)
            
            if verbose and t % 100 == 0:
                print(f"Simulated {str(t)} ms of braintime in {str(time.clock()-t0)} s of computer time.") 
                
        t1 = time.clock()
        print(f"Simulation took {str((t1-t0))} s")
        return voltage_out, recov_out, firings_out
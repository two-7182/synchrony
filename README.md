# Cortical spike synchrony as a measure of contour uniformity
Neurons in the brain cortex can fire synchronously in various situations. This project is modeling synchronous firing in *primary visual cortex*, or *V1 area*. It continues the research of Korndörfer, Ullner et al. (2017), which we describe later.  

The project presentation:   
https://docs.google.com/presentation/d/17un1d3NjZNV0q3btYhq81oOmoSdkYDJsxd7vQC8Jqdw/edit?usp=sharing

The code in this repository helps to create a model, which receives some visual input and responds with a synchronous firing behavior.

## About V1 area
To understand what's going on in the model, we need to remember how V1 area is organized. V1 neurons are specialized. Different neurons look at different regions of the input images. And different neurons can recognize lines of particulat orientation. So, each neuron is looking at some specific region and can recognize the line of some specific orientation in this regions. How does this all look?

- V1 neurons are grouped into structures called **columns**. Neurons of one columns are looking at one specific part of the image. Columns which are next to each other are also looking at the parts of the image next to each other. This is called **retinotopy**.  
- Several columns form one **hypercolumn**. It is processing information from a specific part of the image, and different columns inside this hypercolumn can recognize different angles at this specific part.  

neurons >> columns >> hypercolumns

## Synchrony in V1
Synchrony in V1 is possible due to horizontal connections between neurons (Stettler, Das et al., 2002). The main rule is:
```
Stronger connections lead to a greater synchrony. 
```
The strength of connections depends on two factors:
- input familiarity,
- geometrical characteristics of the input image.

Input familiarity means that neurons have seen the similar input before. This is in focus of Korndörfer et al. (2017). If neurons are familiar with the input, the horizontal connections are stronger. Stronger connections lead to a greater synchrony.

What about the geometrical characteristics? Well, neurons which are spatially close to each other (and also look at the same part of the image - remember retinotopy) have stronger connections. And neurons which respond to similar angles also form stronger connections (Kohn & Smith, 2005).

So, the neurons that are: a) next to each other, b) recognizing specific angles - should be connected more srongly. And this should lead to a better synchrony. 

## Our model
We built a model of V1 area: a neural network, which consists of **Izhikevich** neurons (Izhikevich, 2003). What does it mean?  
The neuron has two intenal variables: `membrane potential` and `recovery variable`. The model operates over time: every time step (e.g. 1 millisecond) several things happen to each neuron:

1. A neuron receives external input + some input from other neurons.
2. It updates its internal variables `membrane potential` and `recovery variable` according to specific formulas.
3. If the value of a `membrane potential` exceeds a certain activation threshold, the neuron produces a spike.
4. After spiking both `membrane potential` and `recovery variable` are reset to initial values.

## Modeling horizontal connections
Wait, but what does this all have to do with horizontal connections?  
Well, remember that each neurons receives the external input and some input from other neurons? The external input comes from the input stimulus. But the input from other neurons is coming to our neuron through the connections.  
Before running the model, we build a connectivity matrix: is specifies the strength of the connection between each pair of neurons. Most of the connections are equal to 0, but neighboring neurons have non-zero connections. The strength of each connection depends on the angle that neurons are recognizing.  

Our model can recognize 4 angles: 0, 45, 90 and 135 degrees. 

## References
* Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on neural networks*, 14(6), 1569-1572.
* Kohn, A., & Smith, M. A. (2005). Stimulus dependence of neuronal correlation in primary visual cortex of the macaque. *Journal of Neuroscience*, 25(14), 3661–3673.
* Korndörfer, C., Ullner, E., García-Ojalvo, J., & Pipa, G. (2017). Cortical spike synchrony as a measure of input familiarity. *Neural computation*, 29(9), 2491-2510.
* Stettler, D. D., Das, A., Bennett, J., & Gilbert, C. D. (2002). Lateral connectivity and contextual interactions in macaque primary visual cortex. *Neuron*, 36(4), 739–750.

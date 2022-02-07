# Cortical spike synchrony as a measure of contour uniformity
Neurons in the brain cortex can fire synchronously in various situations. This project is modeling synchronous firing in *primary visual cortex*, or *V1 area*. It continues the research of Korndörfer et al. (2017), which we describe later.

## More about V1 area
To understand what's going on in the model, we need to remember how V1 area is organized. V1 neurons are specialized. Different neurons look at different regions of the input images. And different neurons can recognize lines of particulat orientation. So, each neuron is looking at some specific region and can recognize the line of some specific orientation in this regions. How does this all look?

V1 neurons are grouped into structures called **columns**. Neurons of one columns are looking at one specific part of the image. Columns which are next to each other are also looking at the parts of the image next to each other. This is called **retinotopy**. Several columns form one **hypercolumn**. It is processing information from a specific part of the image, and different columns inside this hypercolumn can recognize different angles at this specific part.\n
Neurons >> columns >> hypercolumns

## Biological grounds
Synchrony in V1 is possible due to horizontal connections between neurons (Stettler et al., 2002). Stronger connections lead to a greater synchrony. The strength of connections depends on two factors:
- input familiarity,
- geometrical characteristics of the input image.

Input familiarity means that neurons have seen the similar input before. This is in focus of Korndörfer et al. (2017). If neurons are familiar with the input, the horizontal connections are stronger. Stronger connections lead to a greater synchrony.

What about the geometrical characteristics? 



## References
* Stettler, D. D., Das, A., Bennett, J., & Gilbert, C. D. (2002). Lateral connectivity and contextual interactions in macaque primary visual cortex. *Neuron*, 36(4), 739–750.
*  Korndörfer, C., Ullner, E., García-Ojalvo, J., & Pipa, G. (2017). Cortical spike synchrony as a measure of input familiarity. *Neural computation*, 29(9), 2491-2510.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from PIL import Image

img = mpimg.imread('laptop2.png')

def sobell(img):
    """Function for applying sobell filter to images
    args:
        img = numpy array, the picture one wants to change"""
    # 3-dim numpy array, for storing and in the end showing only the edges of the original image
    edges_img = img.copy()
    # rows, columns, dimension, rgb-image
    rows,cols,dim= img.shape
    #print(n,m)

    #create the vertical sobell filter (= to measure gradient in y so vertical direction) by means of a nested list
    sobell_y = [[-1,-2,-1], [0,0,0], [1,2,1]] # both filters are common (nested) list

    #create the horizontal sobell filter (= to measure gradient in x, so horizontal direction) by means of a nested list
    sobell_x = [[-1,0,1], [-2,0,2], [-1,0,1]]

    # move through entire original image, row-wise and column-wise and apply both sobell filters, we have to stop already at n-2 or m-2 respectively in order not to move out of bounds of the image
    for row in range(0, rows-2): #rows-2 not included anymore
        for col in range(0, cols-2):

            #from the original image, create a small 3x3 pixel box which will "move" through the image and by this "catch" all pixels, except for a small border range
            pixels_3x3 = img[row:row+3, col:col+3, 0] #

            #convolve the Sobell mask with the box of 3x3 pixels from the original image in both the y and x-direction
            convolution_y = sobell_y*pixels_3x3
            convolution_x = sobell_x*pixels_3x3

            #in order to map back to values between 0-1 the summed result is divided by 4 -> highest possible value achievable by convolution after summing. This is the rate of change in y direction between neighboring pixels
            gradient_y = convolution_y.sum()/4
            gradient_x = convolution_x.sum()/4

            #overall image gradient for the respective 3x3 box, given by the magnitude of the vector that is made out of x and y values (gradient)
            image_gradient = math.sqrt(gradient_x**2 + gradient_y**2)

            #insert (at respective place) the image gradient into the previously copied image that will only show the edges
            edges_img[row, col] = [image_gradient]*3

    #it might happen that the values are bigger/smaller than 0 and 1 so remap them if needed
    edges_img = edges_img/edges_img.max()
    return edges_img

done_picture = sobell(img) # this done_picture is a numpy array now.
plt.axis("off")
plt.imshow(done_picture)
plt.savefig("plt-sobell-laptop2.png")

# next: turning picture into binary one
img = Image.open('plt-sobell-laptop2.png') #img is instance of class Image. TO DO: One time using matplotlib, the other time PIL, -> one for both??

threshold = 40 #half of 255 #40 for books, 20 for the other 2 pictures (laptop and cube)
decide = lambda pixel_value : 255 if pixel_value > threshold else 0

#L : This image mode has one channel that can take any value between 0 and 255 representing white, black and all the shades of gray in between.
# if we just use it like that we get a grayscale image and if we used img.convert("1") immediately instead we would get the dithered pic
#so we first turn the image into a grayscale image.
# from the docs: in order to use Image.point(lut, mode), and have "mode" have "1", the input mode has to be "L".
grayscale_img = img.convert("L")
binary_img = grayscale_img.point(decide, mode = "1")
binary_img.save('books_sobell-bw.png')
#print(binary_img.mode) # it works. output 1, so only 1 channel, not rgb (or rgba)

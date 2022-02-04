import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

from PIL import Image

class ImagePreprocessor:

    def __init__(self, data_folder, img_name):
        if data_folder[-1] != '/':
            data_folder += '/'
        self.img_name, self.img_ext = img_name.split('.')
        self.data_folder = data_folder
        self.img_path = self.data_folder + self.img_name + '.' + self.img_ext
        self.img = self.upload()
        
    def upload(self):
        return mpimg.imread(self.img_path)
        
    def sobell(self):
        """Function for applying sobell filter to images
        args:
            save = whether or not download filtered image"""
        # 3-dim numpy array, for storing and in the end showing only the edges of the original image
        edges_img = self.img.copy()
        # rows, columns, dimension, rgb-image
        rows, cols, dim = self.img.shape
        #print(n,m)

        #create the vertical sobell filter (= to measure gradient in y so vertical direction) by means of a nested list
        sobell_y = [[-1,-2,-1], [0,0,0], [1,2,1]] # both filters are common (nested) list

        #create the horizontal sobell filter (= to measure gradient in x, so horizontal direction) by means of a nested list
        sobell_x = [[-1,0,1], [-2,0,2], [-1,0,1]]

        # move through entire original image, row-wise and column-wise and apply both sobell filters, we have to stop already at n-2 or m-2 respectively in order not to move out of bounds of the image
        for row in range(0, rows-2): #rows-2 not included anymore
            for col in range(0, cols-2):

                #from the original image, create a small 3x3 pixel box which will "move" through the image and by this "catch" all pixels, except for a small border range
                pixels_3x3 = self.img[row:row+3, col:col+3, 0] #

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
        edges_img = edges_img / edges_img.max()
        
        plt.axis("off")
        plt.imshow(edges_img)
        img_sobel_path = self.data_folder + self.img_name + "_sobell.png"
        plt.savefig(img_sobel_path)  
        return img_sobel_path
        
    def binarize(self, threshold=40, save_sobell=True, save_bw=True):
        img_sobell_path = self.sobell()
        img_sobell = Image.open(img_sobell_path) #img is instance of class Image. TO DO: One time using matplotlib, the other time PIL, -> one for both??
        if save_sobell == False:
            os.remove(img_sobell_path)
        
        decide = lambda pixel_value : 255 if pixel_value > threshold else 0
        #L : This image mode has one channel that can take any value between 0 and 255 representing white, black and all the shades of gray in between.
        # if we just use it like that we get a grayscale image and if we used img.convert("1") immediately instead we would get the dithered pic
        #so we first turn the image into a grayscale image.
        # from the docs: in order to use Image.point(lut, mode), and have "mode" have "1", the input mode has to be "L".
        grayscale_img = img_sobell.convert("L")
        binary_img = grayscale_img.point(decide, mode = "1")
        #print(binary_img.mode) # it works. output 1, so only 1 channel, not rgb (or rgba)
        if save_bw == True:
            binary_path = self.data_folder + self.img_name + "_sobell_bw.png"
            binary_img.save(self.img_path[:-4] + '_sobell-bw.png')
        return binary_img
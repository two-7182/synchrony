import numpy as np
from skimage.draw import line_aa

#drawing lines

def random_line(height, width):
    rr, cc, _ = line_aa(np.random.choice(range(height)), np.random.choice(range(width)), np.random.choice(range(height)), np.random.choice(range(width)))
    return rr, cc

def line_0(height, width):
    h = np.random.choice(range(height))
    rr, cc, _ = line_aa(h, np.random.choice(range(width)), h, np.random.choice(range(width)))
    return rr, cc
    
def line_90(height, width):
    w = np.random.choice(range(width))
    rr, cc, _ = line_aa(np.random.choice(range(height)), w, np.random.choice(range(height)), w)  
    return rr, cc

def line_45(height, width):
    w, h = np.random.choice(range(3,width)), np.random.choice(range(height-3))
    l = np.random.choice(range(2,min(w, height-h)))
    rr, cc, _ = line_aa(h, w, h+l, w-l)  
    return rr, cc

def line_135(height, width):
    w, h = np.random.choice(range(width-3)), np.random.choice(range(height-3))
    l = np.random.choice(range(2,min(width-w, height-h)))
    rr, cc, _ = line_aa(h, w, h+l, w+l)  
    return rr, cc
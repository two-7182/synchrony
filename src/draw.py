import numpy as np
from skimage import draw

#drawing lines

def get_end_coords(start_x, start_y, length, angle):
    if angle == 0:
        end_x, end_y = start_x, start_y + length
    elif angle == 90:
        end_x, end_y = start_x + length, start_y
    elif angle == 45:
        end_x, end_y = start_x + length, start_y + length
    return end_x, end_y

class Line_Square:
    def __init__(self, start_y, start_x, end_y, end_x):
        if end_x <= start_x:
            raise Exception('Please specify start_x < end_x')
        elif end_y <= start_y:
            raise Exception('Please specify start_y < end_y')
        if end_x - start_x != end_y - start_y:
            raise Exception('Please define a square region.')
            
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        
    def draw(self):
        rr, cc = draw.line_nd((self.start_y, self.start_x), (self.end_y, self.end_x))
        return rr, cc

class Line_Square_0(Line_Square):
    def __init__(self, start_y, start_x, end_y, end_x, position='mid'):
        positions_available = ('up', 'down', 'mid')
        if position not in positions_available:
            raise Exception('Please select one of the available line positions: up, down or mid.')
            
        super().__init__(start_y, start_x, end_y, end_x)
        if position == 'mid':
            self.end_y = start_y + int((end_y-start_y) / 2)
            self.start_y = self.end_y
        elif position == 'up':
            self.end_y = start_y
        elif position == 'down':
            self.start_y = end_y
            
class Line_Square_45(Line_Square):
    def __init__(self, start_y, start_x, end_y, end_x, position='mid'):            
        super().__init__(start_y, start_x, end_y, end_x)
        
class Line_Square_90(Line_Square):
    def __init__(self, start_y, start_x, end_y, end_x, position='mid'):
        positions_available = ('left', 'right', 'mid')
        if position not in positions_available:
            raise Exception('Please select one of the available line positions: left, right or mid.')
            
        super().__init__(start_y, start_x, end_y, end_x)
        if position == 'mid':
            self.end_x = start_x + int((end_x-start_x) / 2)
            self.start_x = self.end_x
        elif position == 'left':
            self.end_x = start_x
        elif position == 'right':
            self.start_x = end_x   
            
class Line_Square_135(Line_Square):
    def __init__(self, start_y, start_x, end_y, end_x, position='mid'):            
        super().__init__(start_y, start_x, end_y, end_x)
        self.start_x, self.end_x = end_x, start_x
        
def line_45_joint(strength=1, length=3, width=18, height=18):
    angle_choices = {0: [45], 
                     45: [0, 90],
                     90: [45]}
    img = np.zeros(shape=(height,width))
    
    start_x, start_y = 0, 0
    end_x, end_y = length, length
    last_angle = 0
    
    while start_x < width-length and start_y < height-length: 
        angle = np.random.choice(angle_choices[last_angle])
        end_x, end_y = get_end_coords(start_x, start_y, length, angle)
        
        rr, cc = draw.line_nd((start_y, start_x), (end_y, end_x))
        img[rr, cc] = strength
        
        start_x, start_y = end_x, end_y 
        last_angle = angle
        
    if start_x < width or start_y < height:
        end_x, end_y = width-1, height-1
        rr, cc = draw.line_nd((start_y, start_x), (end_y, end_x))
        img[rr, cc] = strength
    
    return img

def line_45_disjoint(strength=1, length=3, width=18, height=18):
    all_x_start = list(range(0,width,length))
    all_x_end = list(range(length,width,length)) + [width-1]
    all_y_start = list(range(0,height,length))
    all_y_end = list(range(length,height,length)) + [height-1]
    
    angles = {0: {'choice': [45], 'class': Line_Square_0}, 
              45: {'choice': [0, 90], 'class': Line_Square_45},
              90: {'choice': [45], 'class': Line_Square_90},
              -1: {'choice': [0, 45, 90]}}
    img = np.zeros(shape=(height,width))
    last_angle = -1
    
    for start_y, start_x, end_y, end_x in zip(all_y_start, all_x_start, all_y_end, all_x_end):
        angle = np.random.choice(angles[last_angle]['choice'])
        
        rr, cc = angles[angle]['class'](start_y, start_x, end_y, end_x).draw()
        img[rr, cc] = strength
        last_angle = angle

    return img
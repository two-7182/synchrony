import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.draw import line_nd

# creating the different filters

def filter_up_to_44deg(arc, columns=13):

    """function for calculating endpoints of a line, starting at (0/0). When connecting endpoints with startpoints using a
        given angle, the emerging line serves as filter for up to 44°. It decides on the fly how big the filters must be:
        starting with default value of grid size and changing (increasing) it when needed
        args:
        arc: the angle in degrees for which a filter should be created
        columns: needed number of columns which is the filter size. default value is 13. If instead a parameter is used that is too small, the minimal grid size will be calculated
        returns: tuple pairs of start and end points of the line in a numpy array notation and need number of columns for the array (=size)"""

    # this is the degree the line should have, so the angle we want, angles need to be expressed in radians rather than degrees
    angle = math.radians(arc)

    # start points of the line. The vertex of the angle. In a cartesian coordinate system, this would be the point (0/0) -> lower left corner
    x1 =0.0
    y1 =0.0

    rows = int(columns) #square filter

    # right angled triangle: hypotenuse = adjacent (=columns) / cos(angle). Intersects w/ highest column of array
    # adjacent is known => # columns
    hypotenuse = columns / math.cos(angle)

    # use hypotenuse to calculate the end point of it (x2,y2) in the rows x columns grid (array)
    x2 = round(x1 + hypotenuse * math.cos(angle),3) #x-coordinate of the endpoint of the line, actually this is always the # of cols
    y2 = round(y1 + hypotenuse * math.sin(angle),3) # y-coordinate of endpoint of line


    # go from how point annotation in a cartesian coordinate system to how pixels are counted for (in an array)

    #In CC: the numbers get bigger, when "moving up" y-axis
    #In array: number (of rows) get bigger when "moving down" the array.
    y2 = round((rows - y2),3)

    # in order to put it into an array (counting starts at 0), subtract 1 from points so not to get "out of bounds" error
    x2 = round(x2-1,3) #columns in array
    y2 = round(y2-1,3) #rows in array
    # starting points: x1: no need to alter that as it starts always at column 0 -> lower left corner
    y1 = float(columns-1) #just to be consistent turn it into a float

    # only come here if default column parameter isn't used and if one used instead would create too small an array for given angle
    if y2 < 0:
        while y2 <= 0:
            # step for step increase array size
            columns = columns +1
            # set back all previously calculated values
            x1, y1 = 0.0, 0.0
            # do it again
            rows = columns
            hypotenuse = columns / math.cos(angle)
            x2 = round(x1 + hypotenuse * math.cos(angle),3)
            y2 = round(y1 + hypotenuse * math.sin(angle),3)
            y2 = round((columns - y2),3)
            x2 = x2-1
            y2 = round(y2-1,3)
            y1 = float(columns-1) #just to be consistent turn it into a float

    return (x1,y1,x2,y2, columns)     # x1: starting point columns in array, y1: starting points row in array, x2/y2: respective endpoints

def filter_up_to_89deg(arc, rows=13):

    """function for calculating endpoints of a line, starting at (0/0). When connecting endpoints with startpoints using a
        given angle, the emerging line serves as filter for angles between to 46° and 89°.
        It decides on the fly how big the filters must be:
        starting with default value of grid size and changing (increasing) it when needed.
        args:
        arc: the angle in degrees for which a filter should be created
        rows: needed number of rows which is the filter size. default value is 13. If instead a parameter is used that is too small, the minimal grid size will be calculated
        returns: tuple pairs of start and end points of the line in a numpy array notation and need number of columns for the array (=size)"""

    # take the angle between the y-axis and the to be drawn line => 90-given angle
    angle = 90-arc
    angle = math.radians(angle)

    x1 =0.0
    y1 =0.0

    columns = int(rows) #square filter

    #calculate length of hypotenuse which intersects with the highest "rows" of the image pixels
    # known: the length of the adjacent: height of the picture -> the number of rows of array.
    # right angled triangle: hypotenuse = adjacent/cos(angle)
    hypotenuse = rows / math.cos(angle)

    x2 = round(x1 + hypotenuse * math.sin(angle),3)

    # go from point annotation in  CC system to how pixels are counted in array
    # columns are counted same way in cc and arrays. (increasing to the right)
    x2 = round((x2-1),3)
    # the line to be drawn intersects with the rows most highest up: so rows = 0. y denotes rows
    y2 = 0
    # starting points: x1 = columns no need to alter as start always at column 0 -> lower left corner
    y1 = float(columns-1)

    # only if default column parameter isn't used and if the one used instead would create too small an array for given angle
    if x2 < 0:
        while x2 < 0:
            #step for step increase array size
            rows = rows +1
            # set back all previously calculated values
            x1, y1 = 0.0, 0.0
            # do it again
            columns = rows
            hypotenuse = rows / math.cos(angle)
            x2 = round(x1 + hypotenuse * math.sin(angle),3)
            x2 = round((x2-1),3)
            y2 = 0
            y1 = float(columns-1)

    return (x1,y1,x2,y2, rows)     # x1: starting point columns in array, y1: starting points row in array, x2/y2: respective endpoints

def filter_90_to_134deg(arc, rows=13):

    """function for calculating endpoints of a line, starting at (0/0). When connecting endpoints with start points applying a
        given angle, a line can be drawn that serves as filter for angles between 91° and 133°.
        It decides on the fly how big the filters must be:
        starting with default value of grid size and changing (increasing) it when needed
        args: angle: the angle in degrees for which a filter should be created
        returns: tuple pairs of start and end points of the line in a numpy array notation"""


    angle = arc - 90 # angle between y-axis and line: we don't want to use other angles than already used.
    angle = math.radians(angle)

    columns = int(rows)
    x1=0
    y1=0

    # right angled triangle: hypotenuse = adjacent (=rows) / cos(angle). Intersects at highest row of array
    hypotenuse = rows / math.cos(angle)

    # calculate end points of angle line
    x2 = round(x1 + hypotenuse * math.sin(angle),3)

    #array notation
    x2 = round(rows-x2)

    #accomodate need to start counting in arrays at 0
    x2 = round((x2-1),3)
    # endpoint is always row 0
    y2 = 0

    # bring starting points array form, drawing the line from the lower left side (max. # rows, max. # cols).
    x1 = float(columns-1) # array notation: counting starts at 0: accomodate for that
    y1 = x1

    if x2 <= 0:
        while x2 <= 0:
            # increase grid size step by step
            rows = rows +1
            # set back all previously calculated values
            columns = rows
            x1 =0.0
            y1=0.0
            hypotenuse = rows / math.cos(angle)
            x2 = round(x1 + hypotenuse * math.sin(angle),3)
            x2 = round(rows-x2)
            x2 = round((x2-1),3)
            y2 = 0
            x1 = float(columns-1)
            y1 = x1

    return (x1,y1,x2,y2, rows)     # x1: starting point columns in array, y1: starting points row in array, x2/y2: respective endpoints

def filter_greater_135deg(arc, columns=13):

    """function for calculating endpoints of a line, starting at (0/0). When connecting endpoints with start points applying a
        given angle, a line can be drawn that serves as filter for angles between 91° and 134°.
        It decides on the fly how big the filters must be:
        starting with default value of grid size and changing (increasing) it when needed
        args: angle: the angle in degrees for which a filter should be created
        returns: tuple pairs of start and end points of the line in a numpy array notation"""


    angle = 180 - arc # angle between x-axis and line. "drawing angle to the left"
    angle = math.radians(angle)

    rows = int(columns)
    x1=0
    y1=0


    # right angled triangle: hypotenuse = adjacent (=columns) / cos(angle).
    # adjacent is known => # columns
    hypotenuse = columns / math.cos(angle)
    #x-coordinate of the endpoint of the line, actually this is always the # of cols, for angles > 135° always 0:
    #one arm of angle (other one is bottom line (x-axis for CC sys.)) intersects the columns always at 0.
    x2 = 0.0
    y2 = round(y1 + hypotenuse * math.sin(angle),3)
    y2 = round((columns - y2),3)
    y2 = round((y2-1),3)

    # bring starting points array form, drawing the line from the lower left side (max. # rows, max. # cols).
    x1 = float(rows-1) #change from CC notation to array notation: subtract 1 as counting in array starts at 0
    y1 = x1

    # only if default column parameter isn't used and if the one used instead would create too small an array for given angle
    if y2 < 0:
        while y2 < 0:
            # increase grid size step by step
            columns = columns +1
            # set back all previously calculated values
            rows = columns
            x1 =0.0
            y1=0.0
            hypotenuse = columns / math.cos(angle)
            y2 = round(y1 + hypotenuse * math.sin(angle),3)
            y2 = round((columns - y2),3)
            y2 = float(round(y2-1))
            x1 = float(columns-1)
            y1 = x1

    return (x1,y1,x2,y2, columns)  # x1: starting point columns in array, y1: starting points row in array, x2/y2: respective endpoints

# special cases: filters for 0° (180°), 45°, 90°, 135°
def filter_90deg(rows = 13):

    columns = rows
    start_cols = columns-1 #last column in array notation
    start_rows = rows/2
    end_rows = 0
    end_cols = columns/2
    return(start_cols,start_rows,end_cols,end_rows, columns)


def filter_0deg(rows = 13):

    columns = rows
    start_cols = 0
    start_rows = rows-1
    end_rows = rows-1
    end_cols = columns-1
    return(start_rows, start_cols, end_rows, end_cols, columns)


def filter_45deg(rows = 13):

    columns = rows
    start_rows = 0
    start_cols = columns-1
    end_rows = rows-1
    end_cols = 0
    return(start_rows, start_cols, end_rows, end_cols, columns)

def filter_135deg(rows = 13):

    columns = rows
    start_rows = rows-1
    start_cols = columns-1
    end_rows = 0
    end_cols = 0
    return(start_rows, start_cols, end_rows, end_cols, columns)

# function for bringing all the different filters together
def define_filter(arc,columns=13):
    if arc == 0 or arc == 180:
        return(filter_0deg(columns))

    elif arc <= 44:
        return(filter_up_to_44deg(arc, columns))

    elif arc == 45:
        return(filter_90deg(columns))

    elif arc > 45 and arc<90:
        return(filter_up_to_89deg(arc, columns))

    elif arc == 90:
        return(filter_90deg(columns))

    elif arc > 90 and arc<135:
        return(filter_90_to_134deg(arc, columns))

    elif arc == 135:
        return(filter_135deg(columns))

    else: # = arc >135 but <180:
        return(filter_greater_135deg(arc, columns))

# calling the function and depicting the respective filter
start_cols,start_rows,end_cols, end_rows, cols = define_filter(135)
rows = cols
conv_filter = np.zeros(shape=(rows,cols))

angle_line = line_nd((start_rows,start_cols), (end_rows, end_cols), endpoint=True) #using line_nd in order to be able use floats
conv_filter[angle_line] = 1
plt.imshow(conv_filter)
plt.show()

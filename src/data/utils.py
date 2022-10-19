from math import sqrt
from typing import Tuple
import cv2
import numpy as np
import torch
import math

def polar(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    # evaluate reciprocal position
    sc = center(rect_src)
    ec = center(rect_dst)
    new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])) % 360)
    
    if rect_intersect:
        return 0, angle
    elif top and left:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left and bottom:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif bottom and right:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif right and top:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left:
        return (rect_src[0] - rect_dst[2]), angle
    elif right:
        return (rect_dst[0] - rect_src[2]), angle
    elif bottom:
        return (rect_dst[1] - rect_src[3]), angle
    elif top:
        return (rect_src[1] - rect_dst[3]), angle

def transform_image(img_path : str, scale_image=1.0) -> torch.Tensor:
    """ Transform image to torch.Tensor

    Args:
        img_path (str) : where the image is stored
        scale_image (float) : how much scale the image
    """

    np_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    width = int(np_img.shape[1] * scale_image)
    height = int(np_img.shape[0] * scale_image)
    new_size = (width, height)
    np_img = cv2.resize(np_img,new_size)
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    img = img[None,None,:,:]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = 1.0 - img / 128.0
    
    return img

def get_histogram(contents : list) -> list:
    """Create histogram of content given a text.

    Args;
        contents (list)

    Returns:
        list of [x, y, z] - 3-dimension list with float values summing up to 1 where:
            - x is the % of literals inside the text
            - y is the % of numbers inside the text
            - z is the % of other symbols i.e. @, #, .., inside the text
    """
    
    c_histograms = list()

    for token in contents:
        num_symbols = 0 # all
        num_literals = 0 # A, B etc.
        num_figures = 0 # 1, 2, etc.
        num_others = 0 # !, @, etc.
        
        histogram = [0.0000, 0.0000, 0.0000, 0.0000]
        
        for symbol in token.replace(" ", ""):
            if symbol.isalpha():
                num_literals += 1
            elif symbol.isdigit():
                num_figures += 1
            else:
                num_others += 1
            num_symbols += 1

        if num_symbols != 0:
            histogram[0] = num_literals / num_symbols
            histogram[1] = num_figures / num_symbols
            histogram[2] = num_others / num_symbols
            
            # keep sum 1 after truncate
            if sum(histogram) != 1.0:
                diff = 1.0 - sum(histogram)
                m = max(histogram) + diff
                histogram[histogram.index(max(histogram))] = m
        
        # if symbols not recognized at all or empty, sum everything at 1 in the last
        if histogram[0:3] == [0.0,0.0,0.0]:
            histogram[3] = 1.0
        
        c_histograms.append(histogram)
        
    return c_histograms

def to_bin(dist :int, angle : int, b=8) -> torch.Tensor:
    """ Discretize the space into equal "bins": return a distance and angle into a number between 0 and 1.

    Args:
        dist (int): distance in terms of pixel, given by "polar()" util function
        angle (int): angle between 0 and 360, given by "polar()" util function
        b (int): number of bins, MUST be power of 2
    
    Returns:
        torch.Tensor: new distance and angle (binary encoded)

    """
    def isPowerOfTwo(x):
        return (x and (not(x & (x - 1))) )

    # dist
    assert isPowerOfTwo(b)
    m = max(dist) / b
    new_dist = []
    for d in dist:
        bin = int(d / m)
        if bin >= b: bin = b - 1
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_dist.append(bin)
    
    # angle
    amplitude = 360 / b
    new_angle = []
    for a in angle:
        bin = (a - amplitude / 2) 
        bin = int(bin / amplitude)
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_angle.append(bin)

    return torch.cat([torch.tensor(new_dist, dtype=torch.float32), torch.tensor(new_angle, dtype=torch.float32)], dim=1)


from distutils.util import split_quoted
from math import inf, sqrt
import shutil
import cv2
import numpy as np
import torch
import json
import os

from src.paths import NAF


def distance(rectA, rectB):
    """Compute distance from two given bounding boxes
    """
    
    # check relative position
    left = (rectB[2] - rectA[0]) <= 0
    bottom = (rectA[3] - rectB[1]) <= 0
    right = (rectA[2] - rectB[0]) <= 0
    top = (rectB[3] - rectA[1]) <= 0
    
    vp_intersect = (rectA[0] <= rectB[2] and rectB[0] <= rectA[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rectA[1] <= rectB[3] and rectB[1] <= rectA[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 
    
    if rect_intersect:
        return 0
    elif top and left:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[3] - rectA[1])**2))
    elif left and bottom:
        return int(sqrt((rectB[2] - rectA[0])**2 + (rectB[1] - rectA[3])**2))
    elif bottom and right:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[1] - rectA[3])**2))
    elif right and top:
        return int(sqrt((rectB[0] - rectA[2])**2 + (rectB[3] - rectA[1])**2))
    elif left:
        return (rectA[0] - rectB[2])
    elif right:
        return (rectB[0] - rectA[2])
    elif bottom:
        return (rectB[1] - rectA[3])
    elif top:
        return (rectA[1] - rectB[3])
    else: return inf

def transform_image(img_path, scale_image=1.0):

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

def organize_naf(file, name):

    if os.path.isdir(os.path.join(NAF, name)):
        return
    
    os.mkdir(os.path.join(NAF, name))
    org_dir = os.path.join(NAF, name)

    with open(file, 'r') as f:
        train_val_test = json.load(f)

    for key in ['test', 'train', 'valid']:
        split = train_val_test[key]
        org_split = os.path.join(org_dir, key)
        os.mkdir(org_split)
        for group, docs in split.items():
            for doc in docs:
                src = os.path.join(NAF, 'groups', group, doc)
                if not os.path.isfile(src): continue
                dst = os.path.join(org_split, doc)
                shutil.copyfile(src, dst)
                src_ann = os.path.join(NAF, 'groups', group, doc.split(".")[0] + '.json')
                dst_ann = os.path.join(org_split, doc.split(".")[0] + '.json')
                shutil.copyfile(src_ann, dst_ann)

def get_histogram(contents : list) -> list:
    """
    Function
    ----------
    Create histogram of content given a text.

    Parameters
    ----------
    contents : list

    Returns
    ----------
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


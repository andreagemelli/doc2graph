from math import inf, sqrt


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
import numpy as np
import torch
import cv2

def preprocess(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)         #   convert to grayscale, simpler model
    patch = cv2.resize(patch, (32, 32))             #   resizing to a fixed shape helps PCA because PCA can't work for opposite dimensions
    patch = patch.flatten()             #     Flattening the array multilpies the dimensions, since PCA works only on 1D vectors
    return patch

def generate_candidates(frame, x, y, w, h, search_radius = 15, stride = 4):                 #    we get the frame, coordinates of top left corner of drawn rectangle box, width height
    candidates = []             # list of potential candidates
    for dx in range(-search_radius, search_radius + 1, stride):
        for dy in range(-search_radius, search_radius + 1, stride):
            cx = x + dx
            cy = y + dy
            patch = frame[cy : cy+h, cx : cx + w]
            if patch.shape != (h, w, 3):
                continue
            candidates.append([cx, cy, patch])
    
    return candidates


# search radius is how many pixels we use to cast our net, default value is 15
# stride is how many pixels we search by within the search area
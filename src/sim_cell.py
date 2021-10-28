#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 08:44:21 2021

@author: nikolaj
"""

import numpy as np
from numpy.random import uniform, rand, randint
from scipy import interpolate
from skimage.segmentation import flood_fill
from skimage.morphology import binary_dilation
from skimage.filters import gaussian
from skimage.transform import rotate

def cell(size, n = 20, a = .1, b = .15, e = 1, rotang = 0, texture_sigma = 2, edge_sigma = 0):
    """
    

    Parameters
    ----------
    size : TYPE: int
        DESCRIPTION: Size of cell array (will be size x size)
    n : TYPE: int, optional
        DESCRIPTION: Number of points to cell define boundary. The default is 20.
    a : TYPE: double, optional
        DESCRIPTION: Boundary point radius variability. The default is .2.
    b : TYPE: double, optional
        DESCRIPTION: Boundary point angle variability. The default is .1.
    e : TYPE: double, optional
        DESCRIPTION. Kind of eccentricity. The default is 1.
    rotang : TYPE: double, optional
        DESCRIPTION. Angle of cell. Only matters if e != 1. The default is 0.
    texture_sigma : TYPE: double, optional
        DESCRIPTION: Cell interior texture smoothing amount. The default is 2.
    edge_sigma : TYPE: double, optional
        DESCRIPTION: Cell edge smoothing amount. The default is 0.

    Returns
    -------
    c : Type: double array
        DESCRIPTION: Double array of dimensions size x size containing cell

    """
    
    #define cell grid
    c = np.zeros((size, size))
    
    #define cell boundary using eq 2 from Lehmussola 2007
    theta = 2*np.pi*np.linspace(1, n, n)/n
    x = uniform(-a, a, n)+np.cos(theta+uniform(-b, b, n))
    y = (uniform(-a, a, n)+np.sin(theta+uniform(-b, b, n)))
    
    x = np.r_[x, x[0]]
    y = e*np.r_[y, y[0]]
    
    #interpolate cell boundary points
    tck, u = interpolate.splprep([x, y],  s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    
    #normalize cell boundary
    dx = xi.max()-xi.min()
    x0 = (xi.max()+xi.min())/2
    dy = yi.max()-yi.min()
    y0 = (yi.max()+yi.min())/2
    
    xi -= x0
    yi -= y0
    xi /= max(dx, dy)
    yi /= max(dx, dy)
    
    #rasterize curve
    xi = np.int32(xi*(size-1))+size//2
    yi = np.int32(yi*(size-1))+size//2
    
    c[xi, yi] += 1
    c[c>0] = 1
    
    #fill curve
    c = binary_dilation(c) #expand edges so flood fill works
    c = flood_fill(c, (size//2, size//2), 1) #fill cell volume
    
    #create texture
    texture = gaussian(rand(size, size), texture_sigma)
    c = texture*c
    c = gaussian(c, edge_sigma)
    
    if rotang != 0: c = rotate(c, rotang)
    
    return c


def cell_multi(imsize, num_cells, cell_size, n = 10, a = .15, b = .15, e = 1, texture_sigma = 2, edge_sigma = 0, overlap = 0):
    
    #if edge_sigma is not zero you get artifacts where cells overlap
    
    im = np.zeros((imsize, imsize))
    xmax = imsize-max(cell_size)-1
    
    max_tries = 1000
    break_all = False
    
    for i in range(num_cells):
        c = cell(randint(cell_size[0], cell_size[1]), n, a, b, e, rand()*180, texture_sigma, edge_sigma)
        x0, y0 = randint(0, xmax, 2)
        x, y = c.shape
        
        for j in range(max_tries):
            c_mask = c != 0
            im_mask = im[x0:x0+x, y0:y0+y] != 0
            
            if np.mean(c_mask*im_mask) <= overlap:
                #add cell to rest of image
                np.maximum(im[x0:x0+x, y0:y0+y], c, out = im[x0:x0+x, y0:y0+y])
                break
            
            else: x0, y0 = randint(0, xmax, 2)
            
            if j == max_tries: break_all = True
        
        if break_all: break

    return im


    

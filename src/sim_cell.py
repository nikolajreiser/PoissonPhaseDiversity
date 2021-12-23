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
from skimage.morphology import binary_dilation, binary_erosion
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
        DESCRIPTION. Eccentricity (kind of). The default is 1.
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
    y = uniform(-a, a, n)+np.sin(theta+uniform(-b, b, n))
    
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

def cell_binary(size, n = 20, a = .1, b = .15, e = 1, rotang = 0):
    
    #define cell grid
    c = np.zeros((size, size))
    
    #define cell boundary using eq 2 from Lehmussola 2007
    theta = 2*np.pi*np.linspace(1, n, n)/n
    x = uniform(-a, a, n)+np.cos(theta+uniform(-b, b, n))
    y = uniform(-a, a, n)+np.sin(theta+uniform(-b, b, n))
    
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
        
    if rotang != 0: c = rotate(c, rotang)
    
    return c


def cell_multi(imsize, num_cells, cell_size, n = 10, a = .15, b = .15, e = 1, texture_sigma = 2, edge_sigma = 0, overlap = 0):
        
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
            
            if np.mean(c_mask*im_mask) <= overlap: #make sure cells are not overlapping more than specified
                #add cell to rest of image
                np.maximum(im[x0:x0+x, y0:y0+y], c, out = im[x0:x0+x, y0:y0+y])
                break
            
            else: x0, y0 = randint(0, xmax, 2)
            
            if j == max_tries: break_all = True
        
        if break_all: break

    return im


def cell_multi_3D(imsize, num_cells, cell_size, num_layers = 1, n = 10, a = .15, b = .15, e = 1, texture_sigma = 2, edge_sigma = 0, overlap = 0):
    
    #make sure there are odd number of layers
    if num_layers % 2 != 1:
        print("num_layers must be odd number")
        return
    
    im = np.zeros((num_layers, imsize, imsize))
    
    xmax = imsize-max(cell_size)-1
    max_tries = 1000
    break_all = False
    
    #generate individual cells and add to image
    for i in range(num_cells):
        
        #randomly select z midplane of cell        
        z0 = randint(0, num_layers, 1)[0]

        #draw 2D cell (without texture for now)
        c = cell_binary(randint(cell_size[0], cell_size[1]), n, a, b, e, rand()*180)
        
        #create 3D cell and draw 2D cell at selected midplane
        c3d = np.zeros((num_layers, *c.shape))
        c3d[z0] = c
        
        #define how boundary of 2D cell erodes at layers away from cell midplane
        # erosion_matrix = np.ones((3, 3))
        erosion_matrix = None
        
        #create 3D cell by eroding 2D cell boundary
        for j in range(num_layers-z0-1):
            c3d[z0+j+1] = binary_erosion(c3d[z0+j], erosion_matrix)
            
        for j in range(z0):
            c3d[z0-j-1] = binary_erosion(c3d[z0-j], erosion_matrix)


        #choose x and y location for cell
        x0, y0 = randint(0, xmax, 2)
        x, y = c.shape
        c = c3d
        
        
        for j in range(max_tries):
            im_mask = im[:, x0:x0+x, y0:y0+y] != 0

            if np.mean(c*im_mask) <= overlap:  #make sure cells are not overlapping more than specified
                
                #add cell to rest of image
                np.maximum(im[:, x0:x0+x, y0:y0+y], c, out = im[:, x0:x0+x, y0:y0+y])
                    
                break
            
            #if there was too much overlap, find new x and y coordinates to place cell and try again
            else: x0, y0 = randint(0, xmax, 2)
            
            #if too many failed attempts, give up
            if j == max_tries: break_all = True
        
        if break_all: break
    
    #create texture
    texture = gaussian(rand(imsize, imsize), texture_sigma)
    im = texture*im
    im = gaussian(im, edge_sigma)

    return im


    

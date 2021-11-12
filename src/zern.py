#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:30:13 2020

@author: nikolaj
"""
import numpy as np
from scipy.special import comb


def zernfringe2nm(j0, numskip):  #technically fringe-1, so starting at j = 0
    
    j = j0 + numskip + 1
    d = np.floor(np.sqrt(j-1))+1
    
    temp1 = d**2-j
    temp2 = np.mod(temp1, 2)
    m = (1-temp2)*temp1/2 - temp2*(temp1+1)/2
    n = 2*(d-1)-np.abs(m)     
    
    return np.int16(n), np.int16(m)
    
#stolen from https://blog.joey-dumont.ca/zernike-polynomials-coefficients/
def zernnoll2nm(j0, numskip):  #technically noll -1, so starting at j = 0

    j = j0 + numskip + 1

    indices = np.array(np.ceil((1+np.sqrt(1+8*j))/2),dtype=int)-1
    triangular_numbers = np.array(indices*(indices+1)/2).astype(int)
    n = indices -1

    r = j - triangular_numbers
    r +=n
    m = (-1)**j * ((n % 2) + 2 * np.array((r + ((n+1)%2))/2).astype(int))

    return n, m

    
def zernansi2nm(j0, numskip):
    
    j = j0 + numskip
    n = np.uint16(np.ceil((np.sqrt(9+8*j)-3)/2))
    m = 2*j-n*(n+2)
    
    return n, m

def zernnm2ansi(nm, offset):
    n, m = nm
    j = (n*(n+2)+m)/2
    return np.int16(j-offset)

def zernj2nm(j, indexing = "Noll", numskip = 3):
    if indexing == "Noll": return zernnoll2nm(j, numskip)
    if indexing == "ANSI": return zernansi2nm(j, numskip)
    if indexing == "Fringe": return zernfringe2nm(j, numskip)
    if indexing == "Wyant": return zernfringe2nm(j, numskip)
    

def zernfun(j, dim, pupilSize, pixelSize, indexing, rotang, numskip):
    
    r, theta, inds = pupil(dim, pupilSize, pixelSize, rotang)
    n, m = zernj2nm(j, indexing, numskip)

    Rmn = zern_r(n, m, r)
    ang = zern_theta(m, theta)
    
    zern = Rmn*ang
        
    return zern


def zern_theta(m, theta):
   
    if m<0: return np.sin(-m*theta)
    return np.cos(m*theta)


def zern_r(n, m, r):
    
    zum = np.zeros(r.shape)
    mn = int((n-np.abs(m))/2)    
    for k in range(mn+1):
        Rmn = (-1)**k * comb(n-k, k) * comb(n-2*k, mn-k) * r**(n-2*k)
        zum += Rmn
        
    return zum

def grid(dsize, pupilSize, pixelSize, rotang):
    
    d = dsize
    
    pixelSizePhaseY = 1/(d*pixelSize)
    yScale = pixelSizePhaseY/pupilSize
    y0 = d/2-0.5
    yi = np.linspace(-y0, y0, d)*yScale
    
    pixelSizePhaseX = 1/(d*pixelSize)
    xScale = pixelSizePhaseX/pupilSize
    x0 = d/2-0.5
    xi = np.linspace(-x0, x0, d)*xScale
    
    X,Y = np.meshgrid(xi, yi)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y,X)+np.deg2rad(rotang)
    
    return r, theta

def pupil(dim, pupilSize, pixelSize, rotang, p2 = False):
    
    r, theta = grid(dim, pupilSize, pixelSize, rotang)
    P = r<1
    P2 = r>=1.9
    inds = np.nonzero(P)
    inds2 = np.nonzero(P2)
    if p2 == False: return r[inds], theta[inds], inds
    else: return r[inds], theta[inds], inds, inds2

def eps(m): return (m == 0) + 1

def normalize(j, numskip = 3):
    n, m = zernnoll2nm(j, numskip)
    A = (2/eps(m))*(n+1)
    return np.float64(A)

def get_zern(dsize, pupilSize, pixelSize, num_c, rotang = 0, numskip = 3, p2 = False, indexing = "Noll"):
    
    zern = np.array([zernfun(i, dsize, pupilSize, pixelSize, indexing, rotang, numskip) 
                     for i in range(num_c)])
    
    return (zern,) + pupil(dsize, pupilSize, pixelSize, rotang, p2)

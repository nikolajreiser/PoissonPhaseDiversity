#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:34:28 2021

@author: nikolaj
"""

import numpy as np
from deconvolution.functions import fft2, ift2
from numba import njit
from scipy.ndimage.filters import sobel

def rl(ims, s, n_iter = 20, lam = 1e-5, reg_option = 0, reg_steps = 1, b = 0):

    multi = True
    if len(ims.shape) == 2: 
        dim = ims.shape
        multi = False
    if len(ims.shape) == 3:
        dim = ims[0].shape

    u = np.ones(dim)
   
    S = fft2(s)
    St = np.conj(S)
    
    for i in range(n_iter):
        
        if i%reg_steps == 0 and i != 0: reg = get_reg(u, reg_option)
        else: reg = 0
        
        denom = np.abs(ift2(fft2(u)*S))+b
        term1 = fft2(ims/denom)
        
        if multi: factor = (ift2(term1*St)).sum(axis = 0)
        else: factor = (ift2(term1*St))

        u *= factor/(1-lam*reg)
                    
    return u


def get_reg(u, option):
    
    if option == 0: return 0                #no regularizer
    if option == 1: return brl_reg_grad(u)  #bilateral
    if option == 2: return 2*u              #image norm
    if option == 3: return grad_reg(u)      #norm of gradient
    if option == 4: return tv_reg(u)        #total variation

#might be wrong
def grad_reg(u):
    dx, dy = np.gradient(u)
    return 2*np.sqrt(dx**2+dy**2)
    
def tv_reg(u):
    dx, dy = np.gradient(u)
    nrm = np.sqrt(dx**2+dy**2)
    dx = np.divide(dx, nrm, out = np.zeros_like(dx), where = nrm != 0)
    dy = np.divide(dy, nrm, out = np.zeros_like(dy), where = nrm != 0)
    dxx = sobel(dx, axis = 0)
    dyy = sobel(dy, axis = 1)
    return dxx+dyy
    
    
@njit(cache = True)
def brl_reg_grad(im, blur_radius = 30):
    #all radii values should be in units of pixels
    x_l, y_l = im.shape
    reg = np.zeros(im.shape)

    support_radius = blur_radius//2
    sigma_s = (support_radius/3)**2
    sigma_r = np.abs(im.max()-im.min())**2*.01
    
    if sigma_r == 0: return reg

    
    for x in range(x_l):
        for y in range(y_l):
            for r_x in range(support_radius*2):
                for r_y in range(support_radius*2):
                    r_x -= support_radius
                    r_y -= support_radius
                    
                    #skip coordinates if they're outside support radius
                    if np.sqrt(r_x**2+r_y**2) > support_radius: continue
                    
                    dif1 = im[x, y]-im[(x+r_x)%x_l, (y+r_y)%y_l]
                    dif2 = im[x, y]-im[(x-r_x)%x_l, (y-r_y)%y_l]
                    
                    prod1 = (dif1)*g(sigma_r, dif1)
                    prod2 = (dif2)*g(sigma_r, dif2)
                    
                    g_s = g(sigma_s, np.sqrt(r_x**2+r_y**2))
                    reg[x, y] += (g_s/sigma_r)*(prod1+prod2)
                    
    return reg

@njit(cache = True)
def g(sigma, x):
    return np.exp(-(x**2)/(2*sigma))

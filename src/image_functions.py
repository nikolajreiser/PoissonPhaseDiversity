#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:25:27 2020

@author: nikolaj
"""

import numpy as np
import scipy
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pyfftw
from scipy.fftpack import dctn, idctn
from numba import njit
from src.zern import normalize
import os
import pathlib
from sys import stdout

def get_H2(zern, inds, dim, theta, phi):
    
    H = np.zeros(( (len(theta),)+dim ), dtype = np.complex128)

    exp = (phi[:,None]*zern).sum(axis = 0)
    H[:, inds[0], inds[1]] = theta*np.exp(1j*exp)[None,:]
    return H


def get_theta(theta_coef, zern):
    num_theta_coef = theta_coef.shape[1]
    texp = np.sum(theta_coef[:,:,None]*zern[None,:num_theta_coef,:], axis = 1)
    theta = np.exp(1j*texp)
    return theta

def defocus(z, R, inds, NA, l, RI):

    #Z IS IN UNITS OF UM (OR WHATEVER UNITS WAVELENGTH (l) IS IN)
    gamma = np.zeros(len(inds))
    gamma = (2*np.pi*RI/l)*np.sqrt(1-(NA*R/RI)**2)
    
    if type(z) == float:
        return np.exp(1j*z*gamma)
    else:
        exp = z[:, None]*gamma[None,:]*1j
        return np.exp(exp)
    

def sft(im): return np.fft.fftshift(im, axes = (-1, -2))
def fft(im): return scipy.fft.fft2(im)
def ift(im): return scipy.fft.ifft2(im)
def fft2(im): return scipy.fft.rfft2(im)
def ift2(im): return scipy.fft.irfft2(im)




def scl(im,): return np.divide(im-im.min(), im.max()-im.min())

def scl_imgs(imgs, i = True):
    if i:
        return np.divide(imgs-imgs.min(), imgs.max()-imgs.min())
    else:
        return np.divide(imgs-imgs.min(axis = (1,2))[:,None,None], 
                         imgs.max(axis = (1,2))[:,None,None]
                         -imgs.min(axis = (1,2))[:,None,None])
    




@njit(cache = True)
def get_best_var_loc(im, w, ss = 10):

    xs, ys = im.shape
    if xs == w or ys == w: return 0, 0
    xvals = (xs-w)//ss
    yvals = (ys-w)//ss
    
    var = np.zeros((xvals, yvals))
    
    for i in range(xvals):
        for j in range(yvals):
            x = int(i*ss)
            y = int(j*ss)
            var[i, j] = im[x:x+w, y:y+w].var()
            
    amax = var.argmax()
    x = amax//xvals
    y = amax%xvals
    
    return int(x*ss), int(y*ss)


def dcts(im):
    dsize = im.shape[0]
    
    NA = 1.2
    l = .523
    pixelSize = .096
        
    r0 = pixelSize*dsize*2*NA/l
    
    dct = dctn(im)
    dct /= norm(dct)
    dct = np.abs(dct)
    
    x, y = np.meshgrid(np.arange(dsize), np.arange(dsize))
    mask = np.zeros_like(x)
    mask[x+y<r0] = 1
    
    A = -2/(r0**2)
    
    return A*np.sum(dct*np.log2(dct)*mask)


def save_data(data_dir, *argv):
        
    #define path to folder within data folder
    p = pathlib.Path('plotting/data')/data_dir
    
    #create data folder if it does not already exist
    p.mkdir(parents = True, exist_ok = True)
    
    #get list of all files and folders
    dirs = os.listdir(p)
    
    #filter only data_vXX folders
    dirs = [d for d in dirs if d[:6] == 'data_v']
    
    #get highest version folder
    dir_nums = [int(d.split('v')[1]) for d in dirs]
    if not dir_nums: dir_num_max = 0 #check if dir_nums is empty
    else: dir_num_max = max(dir_nums)
    
    #create new folder with one higher version
    folder = f'data_v{dir_num_max+1}'
    p = p/folder
    p.mkdir(parents = True)
    
    for i, d in enumerate(argv):
        np.save(p/f'd{i}.npy', d)
    
def load_data(data_dir, v = -1):
    
    #define path to folder within data folder
    p = pathlib.Path('plotting/data')/data_dir

    #get list of all files and folders
    dirs = os.listdir(p)
    
    #filter only data_vXX folders
    dirs = [d for d in dirs if d[:6] == 'data_v']
    
    #get highest version folder (or folder corresponding to v)
    dir_nums = [int(d.split('v')[1]) for d in dirs]

    if v == -1: dir_num = max(dir_nums)
    else: dir_num = v
    
    #find path corresponding to folder
    folder = f'data_v{dir_num}'
    p = p/folder
    
    data_files = os.listdir(p)
    data_files = sorted(data_files)
    datas = []
    for d in data_files:
        datas.append(np.load(p/d))
    
    return datas

def imshow(im, cbar = False, vmin = None, vmax = None, cmap = 'viridis', cbar_name = None, hide_ax = True):
    
    
    plt.imshow(im, cmap = cmap, vmin = vmin, vmax = vmax, interpolation = 'none')
    if hide_ax: plt.axis('off')
    if cbar: plt.colorbar(label = cbar_name)
    plt.show()
    
def err(c, phi, show = True):
    err_val = 100*norm(c-phi)/norm(phi)
    if show: print(f"Error: {err_val:.2f}%")
    return err_val

def errA(c, phi, show = True):
    #assuming c and phi are in radians
    
    A = normalize(np.arange(len(c)))
    dif = c-phi
    # rms_aberrated = np.sqrt(np.mean(A*phi**2))/(2*np.pi)
    rms_corrected = np.sqrt(np.sum(A*dif**2))/(2*np.pi)
    
    if show:
        # print(f"Aberrated Wavefront RMS: {rms_aberrated:.2f} waves")
        print(f"Corrected Wavefront RMS: {rms_corrected:.2f} waves")
    return rms_corrected

def progress(string):
    stdout.write("\r"+string)
    stdout.flush()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""


import numpy as np
from numpy.random import rand, seed
from numpy.linalg import norm
from plotting import imshow, metric_loc, err, progress, bar_plot_2, bar_plot, strehl, errA
from iterate_poisson import iter_p
from iterate_gaussian import iter_g, iter_g2, iter_g0, iter_g3, iter_g_lin
from sim_im import load_imgs
from zern import get_zern
import matplotlib.pyplot as plt
from image_functions import get_theta, scl_imgs
from fast_fft import Fast_FFTs
import multiprocessing as mp
from skimage.transform import downscale_local_mean as dlm

#constants and conversion
pixelSize = .096
NA = 1.2
l = .532
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase
ansi2noll = np.array([4, 3, 5, 7, 8, 6, 9, 12, 13, 11, 14, 10]) - 1
signs = -1*np.array([0, 0, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1])

rotang = -70
num_c = 12
imsize = 256
dsize = 256
# x0 = 460
# y0 = 440
x0 = 373
y0 = 373
scale = dsize//imsize

# x0 = 200
# y0 = 300

imgs_idx_1 = [0, 1, 2]
num_imgs_1 = len(imgs_idx_1)

imgs_idx_2 = [0, 1, 2, 3, 4]
num_imgs_2 = len(imgs_idx_2)

dim = (imsize, imsize)

#precompute zernike polynomials and ffts
zern, R, Theta, inds = get_zern(dsize, pupilSize, pixelSize*scale, num_c, rotang = rotang)
ff_1 = Fast_FFTs(dsize, num_imgs_1, 1)
ff_2 = Fast_FFTs(dsize, num_imgs_2, 1)

#aggregate data
nd = 50
srp1 = np.zeros((nd))
srg1 = np.zeros((nd))
srp2 = np.zeros((nd))
srg2 = np.zeros((nd))
sri = np.zeros((nd))

show = False

for i in range(nd):
    progress(f'{i+1}/{nd}')
    
    # Load data
    dataset = str(i)
    datadir = '/home/nikolaj/Downloads/20200820_Beads_PR/Progressive_batch_50/'
    coeffs = np.loadtxt(datadir+'beads_'+dataset+'_coeff.txt')
    
    imgs = load_imgs(datadir+'beads_'+dataset+'.tif')
    imgs = np.float64(imgs)
    
    imgs = imgs[:,x0:x0+imsize, y0:y0+imsize]
    imgs = dlm(imgs, (1, scale, scale))
    
    # imshow(imgs[0])
    # import sys
    # sys.exit()
    
    imgs -= 300
    imgs[imgs<0] = 0
    imgs = scl_imgs(imgs)
    
    imgs1 = imgs[imgs_idx_1]
    imgs2 = imgs[imgs_idx_2]
    
    coeffs = coeffs[:,:len(signs)]*signs
    coeffs = coeffs[:,[ansi2noll]][:,0,:]
    coeffs *= l2p
    
    theta01 = np.zeros((num_imgs_1, num_c))
    theta01[1:] = coeffs[imgs_idx_1[1:]]
    theta1 = get_theta(theta01, zern)
    
    theta02 = np.zeros((num_imgs_2, num_c))
    theta02[1:] = coeffs[imgs_idx_2[1:]]
    theta2 = get_theta(theta02, zern)

    phi = coeffs[0]
    c0 = np.zeros((num_c))+1e-10
        
    c11 = iter_g0(zern, inds, imgs1, theta1, Sk0, c0, ff_1, 10, show)[0]
    c21 = iter_p(zern, inds, imgs1, theta1, Sk0, c0, ff_1, phi, imgs1[0], show)[0].copy()
    
    # c12, wob1, g_cist = iter_g0(zern, inds, imgs2, theta2, Sk0, c0, ff_2, 10, show)
    # c22 = iter_p(zern, inds, imgs2, theta2, Sk0, c0, ff_2, phi, imgs2[0], show)[0].copy()
    
    sri[i] = errA(c11*0, phi, show = False)
    srg1[i] = errA(c11, phi, show = False)
    srp1[i] = errA(c21, phi, show = False)
    
    # srg2[i] = errA(c12, phi, show = False)
    # srp2[i] = errA(c22, phi, show = False)
    
xs = np.argsort(sri)

fig, ax = plt.subplots()


ax.plot(sri[xs], srg1[xs], c = 'orange', label = "Gaussian")
ax.plot(sri[xs], srp1[xs], c = 'blue', label = "Poisson")
ax.plot(sri[xs], sri[xs], c = 'black', ls = '--', label = "Initial Aberration")

# ax.plot(sri[xs], srg2[xs], c = 'orange', ls = '--', label = "Gaussian (5 Images)")
# ax.plot(sri[xs], srp2[xs], c = 'blue', ls = '--', label = "Poisson (5 Images)")


ax.set_ylabel(r"RWE ($\lambda$)")
ax.set_xlabel(r"Initial Wavefront Magnitude ($\lambda$)")
fig.legend(loc = 'upper left', bbox_to_anchor=(.12, .88))
fig.show()

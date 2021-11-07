#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""


import numpy as np
from numpy.random import rand, seed
from numpy.linalg import norm
from src.iterate_poisson import iter_p
from src.iterate_gaussian import iter_g0
from src.sim_im import sim_im, sim_im_2, sim_im_3, add_noise
from src.zern import get_zern
import matplotlib.pyplot as plt
from src.image_functions import get_theta, scl, ift, sft, defocus, errA, imshow
from src.fast_fft import Fast_FFTs
from src.zern import normalize
from src.sim_cell import cell_multi
from skimage.transform import downscale_local_mean as dlm


#precompute pupil and zernike polynomials
pixelSize = .096
NA = 1.2
l = .532
RI = 1.33
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase

dsize = 512
dim0 = (dsize, dsize)
num_imgs = 3
num_phi = 42
rotang = 0

seed(1)

imsize = 256
num_c = 12

num_theta = num_c
div_mag = 2 #for defocus, this is distance from focal plane in waves
div_mag *= l #convert waves to um
# div_mag *= l2p*NA**2/(4*RI) #convert um to radians. Extra factor of NA^2/4RI is because of Zernike defocus approximation (see 10.1364/JOSAA.37.000016)
theta0 = np.zeros((num_imgs, num_theta))


theta0[0,0] = div_mag
theta0[1,0] = -div_mag


# theta0[0,1] = div_mag
# theta0[1,1] = -div_mag
# theta0[2,2] = div_mag
# theta0[3,2] = -div_mag

# theta0[4,0] = div_mag/4
# theta0[5,0] = -div_mag/4


# theta0[0,1] = div_mag*np.cos(2*np.deg2rad(0))
# theta0[0,2] = -div_mag*np.sin(2*np.deg2rad(0))
# theta0[1,1] = div_mag*np.cos(2*np.deg2rad(60))
# theta0[1,2] = -div_mag*np.sin(2*np.deg2rad(60))
# theta0[2,1] = div_mag*np.cos(2*np.deg2rad(120))
# theta0[2,2] = -div_mag*np.sin(2*np.deg2rad(120))


zern0, R0, Theta, inds0, = get_zern(dsize, pupilSize, pixelSize, num_phi)
theta = get_theta(theta0, zern0)
theta = defocus(np.array([-div_mag, div_mag, 0]), R0, inds0, NA, l, RI)
dim = (imsize,imsize)

zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*2, num_c)
theta1 = get_theta(theta0, zern[:num_c])
theta1 = defocus(np.array([-div_mag, div_mag, 0]), R, inds, NA, l, RI)

ff = Fast_FFTs(imsize, num_imgs, 1)


# ob = cell_multi(dsize*2, 300, (10, 60), e = .7, overlap = .2)
ob = cell_multi(dsize*2, 400, (30, 60), e = .7, overlap = .05)
# ob_1024 = cell_multi(1024*2, 1600, (30, 60), e = .7, overlap = .05)

# ob = dlm(ob_1024, (2,2))
# ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
# ob = cell_multi(dsize*2, 100, (10, 15), e = 1, overlap = .5)

# ob = cell_multi(dsize//2, 10, (11, 12),  a = 0, b = 0, e = 1, texture_sigma = 5, edge_sigma = 0, overlap = .5)
# w = int(dsize*3/4)
# ob = np.pad(ob, ((w, w), (w, w)))
show = True
abmag0 = 2
abmag1 = abmag0/2

phi0 = (rand(num_c)*2-1)
phi0 /= norm(phi0*np.sqrt(normalize(np.arange(num_c))))
phi0 *= abmag0

phi1 = (rand(num_phi-num_c)*2-1)
if num_c != num_phi:    
    phi1 /= norm(phi1*np.sqrt(normalize(np.arange(num_c, num_phi))))
    phi1 *= abmag1

phi = np.zeros((num_phi))
phi[:num_c] = phi0
phi[num_c:] = phi1

phi *= 2*np.pi #convert waves to radians
num_photons = np.ones((num_imgs, 1, 1))

pb = 500
pf = 1
num_photons *= pb/pf
num_photons[-1] *= pf


imgs0 = sim_im_2(ob, dim0, phi, num_imgs, theta, zern0, R0, inds0)
# imgs0 = sim_im(ob, dim0, phi, num_imgs, theta, zern0, R0, inds0)
# imgs0 = sim_im_3d_2(ob, dim0, phi, num_imgs, theta, zern0, R0, inds0)
# imgs0 = sim_im_3d_1(ob, dim0, phi, num_imgs, theta, zern0, R0, inds0)
# snr, imgs0 = add_noise(imgs0, num_photons = num_photons, dark_noise = 100, read_noise = 20)
snr, imgs0 = add_noise(imgs0, num_photons = num_photons, dark_noise = 1, read_noise = 2)
# imgs0[-1] /= pf
# print(f"SNR: {snr:.2f}")

# imshow(imgs0[-1])
imshow(np.reshape(imgs0, ((dsize*num_imgs, dsize))).T)

imgs = dlm(imgs0, (1, 2, 2))
# imshow(np.reshape(imgs, ((imsize*num_imgs, imsize))).T)

# if imsize == dsize: imgs = imgs0
imgs = scl(imgs)

# else:  
#     lx, ly = metric_loc(imgs0, imsize, 16)
#     x0 = lx
#     y0 = ly
#     imgs = imgs0[:,x0:x0+imsize,y0:y0+imsize]

# imshow(imgs[0])
# imshow(imgs[-1][10:-10, 10:-10])

c0 = np.zeros((num_c))+1e-10

c1, wob1, g_cist = iter_g0(zern[:num_c], inds, imgs, theta1, Sk0, c0, ff, 100, show)
eg = errA(c1, phi0*2*np.pi, show)
c2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs, theta1, Sk0, c0.copy(), ff, show)

# c2, cost2, num_iter2, sss = iter_p2(zern[:num_c], inds, imgs, theta1, Sk0, c0.copy(), ff, show)

ep = errA(c2, phi0*2*np.pi, show)


# bar_plot_2(c1/(2*np.pi), c2/(2*np.pi), phi/(2*np.pi), 4)

# imshow(ob[256:-256, 256:-256])

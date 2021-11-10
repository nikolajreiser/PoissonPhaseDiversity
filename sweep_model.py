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
from src.sim_im import sim_im, sim_im_2, add_noise
from src.zern import get_zern, normalize
from src.image_functions import scl_imgs, defocus, save_data, progress, errA
from src.fast_fft import Fast_FFTs
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

seed(4)

imsize = 256
num_c = 12

num_theta = num_c
div_mag = 1
div_mag *= l #convert from waves to um

zern, R, Theta, inds = get_zern(imsize, pupilSize*2, pixelSize, num_c)
zern0, R0, Theta, inds0, = get_zern(dsize, pupilSize, pixelSize, num_phi)

theta = defocus(np.array([-div_mag, div_mag, 0]), R, inds, NA, l, RI)
theta0 = defocus(np.array([-div_mag, div_mag, 0]), R0, inds0, NA, l, RI)

dim = (imsize,imsize)

ff = Fast_FFTs(imsize, num_imgs, 1)

# ob = cell_multi(dsize*2, 300, (10, 60), e = .7, overlap = .2)
# obname = "ob1"

ob = cell_multi(dsize*2, 400, (30, 60), e = .7, overlap = .05)
obname = "ob2"

# ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
# obname = "ob3"

# ob = cell_multi(dsize*2, 100, (10, 15), e = 1, overlap = .5)
# obname = "ob4"

show = True
abmag0 = 2
abmag1 = abmag0/2

num_photons = 500
dark_noise = 1
read_noise = 2

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

phi_low_ord = phi.copy()
phi_low_ord[num_c:] = 0

imgs0 = sim_im(ob, dim0, phi_low_ord, num_imgs, theta0, zern0, R0, inds0)
imgs1 = sim_im(ob, dim0, phi, num_imgs, theta0, zern0, R0, inds0)
imgs2 = sim_im_2(ob, dim0, phi_low_ord, num_imgs, theta0, zern0, R0, inds0)
imgs3 = sim_im_2(ob, dim0, phi, num_imgs, theta0, zern0, R0, inds0)

snr, imgs0 = add_noise(imgs0, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)
snr, imgs1 = add_noise(imgs1, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)
snr, imgs2 = add_noise(imgs2, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)
snr, imgs3 = add_noise(imgs3, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)


imgs0 = dlm(imgs0, (1, 2, 2))
imgs1 = dlm(imgs1, (1, 2, 2))
imgs2 = dlm(imgs2, (1, 2, 2))
imgs3 = dlm(imgs3, (1, 2, 2))
    
imgs0 = scl_imgs(imgs0)
imgs1 = scl_imgs(imgs1)
imgs2 = scl_imgs(imgs2)
imgs3 = scl_imgs(imgs3)

c0 = np.zeros((num_c))+1e-10

err_g = np.zeros((4))
err_p = np.zeros((4))

cg = iter_g0(zern, inds, imgs0, theta, Sk0, c0, ff, 100, show)[0]
cp = iter_p(zern, inds, imgs0, theta, Sk0, c0.copy(), ff, show)[0]
err_g[0] = errA(cg, phi0*2*np.pi, show)
err_p[0] = errA(cp, phi0*2*np.pi, show)

cg = iter_g0(zern, inds, imgs1, theta, Sk0, c0, ff, 100, show)[0]
cp = iter_p(zern, inds, imgs1, theta, Sk0, c0.copy(), ff, show)[0]
err_g[1] = errA(cg, phi0*2*np.pi, show)
err_p[1] = errA(cp, phi0*2*np.pi, show)

cg = iter_g0(zern, inds, imgs2, theta, Sk0, c0, ff, 100, show)[0]
cp = iter_p(zern, inds, imgs2, theta, Sk0, c0.copy(), ff, show)[0]
err_g[2] = errA(cg, phi0*2*np.pi, show)
err_p[2] = errA(cp, phi0*2*np.pi, show)

cg = iter_g0(zern, inds, imgs3, theta, Sk0, c0, ff, 100, show)[0]
cp = iter_p(zern, inds, imgs3, theta, Sk0, c0.copy(), ff,show)[0]
err_g[3] = errA(cg, phi0*2*np.pi, show)
err_p[3] = errA(cp, phi0*2*np.pi, show)

save_data('model_test', err_g, err_p)
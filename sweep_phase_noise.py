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
from src.image_functions import scl_imgs, defocus, save_data, progress, errA, get_theta
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

seed(0)

imsize = 256
num_c = 12
num_theta = num_c


zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*2, num_phi)
zern0, R0, Theta, inds0, = get_zern(dsize, pupilSize, pixelSize, num_phi)

dim = (imsize,imsize)
ff = Fast_FFTs(imsize, num_imgs, 1)


div_mag = 6


# ob = cell_multi(dsize*2, 300, (10, 60), e = .7, overlap = .2)
# ob = cell_multi(dsize*2, 400, (30, 60), e = .7, overlap = .05)
ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
# ob = cell_multi(dsize*2, 100, (10, 15), e = 1, overlap = .5)




show = False


theta = defocus(np.array([-div_mag, div_mag, 0])/l, R0, inds0)

phase_mags = np.linspace(0, 2, 11)
num_points = 100
errs_g = np.zeros((len(phase_mags), num_points))
errs_p = np.zeros((len(phase_mags), num_points))

num_photons = 500
dark_noise = 1
read_noise = 2

abmag0 = 2
abmag1 = abmag0/2

for i, phase_mag in enumerate(phase_mags):
    for j in range(num_points):
        progress(f"{i} \t {j}")

        
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
        
        phase_aber = rand(num_imgs, num_phi)*2 - 1
        for k in range(num_imgs):
            phase_aber[k] /= norm(phase_aber[k]*np.sqrt(normalize(np.arange(num_phi))))
        phase_aber *= phase_mag*2*np.pi
        theta_noise = get_theta(phase_aber, zern)

        imgs0 = sim_im_2(ob, dim0, phi, num_imgs, theta*theta_noise, zern0, R0, inds0)
        snr, imgs = add_noise(imgs0, num_photons = num_photons, dark_noise = dark_noise, read_noise = read_noise)
        imgs = dlm(imgs, (1, 2, 2))
        imgs = scl_imgs(imgs)

        if show: print(f"SNR: {snr:.2f}")
        
        phi = phi0[:num_c]
        
        
        c0 = np.zeros((num_c))+1e-10
        ob_temp = ob[128:-128, 128:-128]
        
        c1, wob1, g_cist = iter_g0(zern[:num_c], inds, imgs, theta, Sk0, c0, ff, 100, show)
        errs_g[i, j] = errA(c1, phi0*2*np.pi, show)
        c2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs, theta, Sk0, c0.copy(), ff, show)
        errs_p[i, j] = errA(c2, phi0*2*np.pi, show)  


save_data('sweep_phase_noise', phase_mags, errs_g, errs_p)
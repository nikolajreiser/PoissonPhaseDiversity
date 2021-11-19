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

seed(1)

imsize = 256
num_c = 12
num_theta = num_c


zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize*2, num_phi)
dim = (imsize,imsize)
ff = Fast_FFTs(imsize, num_imgs, 1)
ff_2 = Fast_FFTs(imsize, 2, 1)


# ob = cell_multi(dsize*2, 300, (10, 60), e = .7, overlap = .2)
# obname = "ob1"

# ob = cell_multi(dsize*2, 400, (30, 60), e = .7, overlap = .05)
# obname = "ob2"

# ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
# obname = "ob3"

ob = cell_multi(dsize*2, 100, (10, 15), e = 1, overlap = .5)
obname = "ob4"

show = False
print(obname)



div_mags = np.linspace(10, 0, 20, False) #in units of waves
div_mags *= l #convert waves to um
num_points = 100

errs_g = np.zeros((len(div_mags), num_points))
errs_p = np.zeros((len(div_mags), num_points))

errs_g_2 = np.zeros((len(div_mags), num_points))
errs_p_2 = np.zeros((len(div_mags), num_points))

abmag0 = 4
abmag1 = abmag0/2



for i, div_mag in enumerate(div_mags):
    for j in range(num_points):
        progress(f"{i+1}/{len(div_mags)} \t {j+1}/{num_points}")

        theta = defocus(np.array([-div_mag, div_mag, 0]), R, inds, NA, l, RI)
        
        
        show = False
        
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
        
        
        imgs = sim_im_2(ob, dim0, phi, num_imgs, theta, zern, R, inds)
        # snr, imgs = add_noise(imgs, num_photons = 1000, dark_noise = 50, read_noise = 10)
                
        imgs = dlm(imgs, (1, 2, 2))
        imgs = scl_imgs(imgs)
        
        c0 = np.zeros((num_c))+1e-10
        ob_temp = ob[128:-128, 128:-128]
        
        c1, wob1, g_cist = iter_g0(zern[:num_c], inds, imgs, theta, Sk0, c0, ff, 100, show)
        errs_g[i, j] = errA(c1, phi0*2*np.pi, show)
        c2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs, theta, Sk0, c0.copy(), ff, show)
        errs_p[i, j] = errA(c2, phi0*2*np.pi, show)
        
        c1_2, wob1, g_cist = iter_g0(zern[:num_c], inds, imgs[:-1], theta[:-1], Sk0, c0, ff_2, 100, show)
        errs_g_2[i, j] = errA(c1_2, phi0*2*np.pi, show)
        c2_2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs[:-1], theta[:-1], Sk0, c0.copy(), ff_2, show)
        errs_p_2[i, j] = errA(c2_2, phi0*2*np.pi, show)

save_data(f'sweep_diversity_2v3_{obname}_{abmag0}', div_mags, errs_g, errs_p, errs_g_2, errs_p_2)




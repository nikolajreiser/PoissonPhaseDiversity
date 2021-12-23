#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:03:32 2020

@author: nikolaj
"""

import numpy as np
from numpy.random import rand, seed
from numpy.linalg import norm
from src.iterate_poisson import iter_p
from src.sim_im import sim_im, sim_im_2, add_noise
from src.zern import get_zern, normalize
import matplotlib.pyplot as plt
from src.image_functions import fft2, sft, get_H2, ift, ift2, scl_imgs, errA, defocus
from src.fast_fft import Fast_FFTs
from src.sim_cell import cell_multi

from cuda.c_functions_py import CuFunc

#precompute pupil and zernike polynomials
pixelSize = .096
NA = 1.2
l = .532
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase
dsize = 256
dim = (dsize, dsize)
num_imgs = 3
num_phi = 12
rotang = 0

zern, R, Theta, inds = get_zern(dsize, pupilSize, pixelSize, num_phi)
num_inds = len(inds[0])
cu_inds = np.int32(inds[0]*dsize+inds[1])
ff = Fast_FFTs(dsize, num_imgs, 1)


cf = CuFunc(num_imgs, dsize, num_phi)
d_zern = cf.load_array(np.float32(zern))
d_inds = cf.load_array(cu_inds)

hsize = dsize//2 + 1
l2d = dsize*dsize
h2d = hsize*dsize
l3d = num_imgs*l2d
h3d = num_imgs*h2d

d_wavefront = cf.cu_init_array(num_inds, 0)
d_H = cf.cu_init_array(l3d, 1)
d_h = cf.cu_init_array(l3d, 1)
d_s = cf.cu_init_array(l3d, 0)
d_S = cf.cu_init_array(h3d, 1)
d_G = cf.cu_init_array(h3d, 1)
d_g = cf.cu_init_array(l3d, 0)
h_g = cf.c_init_array(l3d, 0)

r2c = cf.cu_init_r2c(num_imgs, dsize)
c2c = cf.cu_init_c2c(num_imgs, dsize)
c2r = cf.cu_init_c2r(num_imgs, dsize)
r2cs = cf.cu_init_r2cs(dsize)
c2rs = cf.cu_init_c2rs(dsize)

handle = cf.cu_init_cublas()

seed(3)

phi = (rand(num_phi)*2-1)*1
phi /= np.linalg.norm(phi)
phi *= 4


ob = cell_multi(dsize*2, 1000, (30, 60), e = .1, overlap = .5)
hob = ob[dsize//2:-dsize//2, dsize//2:-dsize//2]
hob = np.ascontiguousarray(hob)
F = fft2(hob)
show = True

theta0 = np.zeros((num_imgs, num_phi))
RI = 1.33
div_mag = 3
div_mag *= l
theta = defocus(np.array([-div_mag, div_mag, 0]), R, inds, NA, l, RI)


# d_phi = cf.load_array(phi)
d_theta = cf.load_array(np.complex64(theta))
d_F = cf.load_array(np.complex64(F))

imgs0 = sim_im_2(ob, dim, phi, num_imgs, theta, zern, R, inds)
snr, imgs0 = add_noise(imgs0, num_photons = 500, dark_noise = 1, read_noise = 2)
imgs0 = scl_imgs(imgs0)

f = np.ones((dsize, dsize))

num_updates = 1
num_c = 12
zern = zern[:num_c]
c0 = np.zeros((num_c))+1e-10
# c0 = np.random.rand(num_c)
c2, cost2, num_iter2, sss = iter_p(zern[:num_c], inds, imgs0, theta, Sk0, c0.copy(), ff, show)

d_c = cf.load_array(np.float32(c0))
dU = normalize(np.arange(num_c))*np.pi/Sk0
d_dU = cf.c_load_array(num_c, np.float32(dU))
d_dF = cf.cu_init_array(h2d, 1)
d_df = cf.cu_init_array(l2d, 0)
d_Q = cf.cu_init_array(h3d, 1)
d_f = cf.load_array(np.float32(f))
d_q = cf.cu_init_array(l3d, 0)
d_inner = cf.cu_init_array(l3d, 0)
d_inner1 = cf.cu_init_array(l3d, 1)
d_dL_integrand = cf.cu_init_array(l3d, 0)
d_dc = cf.c_init_array(num_c, 0)
d_imgs = cf.load_array(np.float32(imgs0))
d_G = cf.cu_init_array(h3d, 1)


cf.cu_ipc(num_imgs, dsize, num_c, num_inds, num_updates, d_F, d_theta, d_c, d_zern, d_wavefront, 
          d_H, d_h, d_s, d_S, d_G, d_g, r2c, c2r, c2c, r2cs, c2rs, handle, d_inds, h_g, d_dU, d_dF, d_Q, d_f, d_df,
          d_q, d_inner, d_inner1, d_dL_integrand, d_dc, d_imgs, np.float32(num_imgs*Sk0))

# ftest = cf.cu_unload_array(num_inds, d_dL_integrand)

# i = cf.cu_unload_array2(l3d, d_inner)
# q = cf.cu_unload_array2(l3d, d_q)
c1 = cf.cu_unload_array3(num_c, d_c)


# F = cf.cu_unload_array_complex3(h2d, d_F)
# S = cf.cu_unload_array_complex2(h3d, d_S)
# i1 = cf.cu_unload_array_complex(l3d, d_inner1)
H = cf.cu_unload_array_complex(l3d, d_H)
# G = cf.cu_unload_array_complex2(h3d, d_G)
# Q = cf.cu_unload_array_complex2(h3d, d_Q)

errA(c1, phi)
errA(c2, phi)
errA(c1, c2)


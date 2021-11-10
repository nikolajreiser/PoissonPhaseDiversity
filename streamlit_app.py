#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""

import streamlit as st
import numpy as np
from src.iterate_poisson import iter_p
from src.iterate_gaussian import iter_g0
from src.sim_im import sim_im, sim_im_2, add_noise
from src.zern import get_zern
import matplotlib.pyplot as plt
from src.image_functions import get_theta, scl_imgs, ift, sft, defocus
from src.fast_fft import Fast_FFTs
from src.zern import normalize
from src.zern import zernnoll2nm as zn2nm
from src.sim_cell import cell_multi
from src.streamlit_functions import run_estimation
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")
st.title('Phase Diversity')

link = '[Usage and code on github](http://github.com/nikolajreiser/PoissonPhaseDiversity)'
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.write("When closing and opening expanders, the numeric values inside get reset because of a streamlit bug so be careful")

with st.sidebar.expander(label = "Microscope Settings"):
    NA = st.number_input(label = "NA", min_value = 0.0, value  = 1.2, step = 0.1)
    pixelSize = st.number_input(label = "Pixel Size (um)", format = "%.3f", min_value = 0.0, value  = .096)
    l = st.number_input(label = "Wavelength (um)", format = "%.3f", min_value = 0.0, value  = .532)
    RI = st.number_input(label = "Refractive Index", min_value = 0.0, value  = 1.33)
    rotang = st.number_input(label = "Pupil Angle Offset (degrees)", value  = 0.0)
    
with st.sidebar.expander(label = "Algorithm Settings"):
    num_c = int(st.number_input(label = "# of orders to estimate", min_value = 2, step = 1, value = 12))
    num_imgs = int(st.number_input(label = "# of images", min_value = 2, step = 1, value = 3))
    units = st.radio(label = "Aberration Units", options = ("um", "waves", "radians"))
    algo = st.radio(label = "Algorithm Type:", options = ("Poisson", "Gaussian"))
    div_type = st.radio(label = "Diversity Phase Type:", options = ("Defocus", "Zernike Polynomials"))
    if div_type == "Zernike Polynomials":
        indexing = st.radio(label = "Zernike Indexing:", options = ("Noll", "ANSI", "Fringe", "Wyant"))

num_phases = [None for i in range(num_imgs)]
use_im = [None for i in range(num_imgs)]
imgs = [None for i in range(num_imgs)]
defocuses = [None for i in range(num_imgs)]
zern_idx = [None for i in range(num_imgs)]
zern_str = [None for i in range(num_imgs)]
if 'cs' not in st.session_state:
    st.session_state.cs = []

for i in range(num_imgs):
    
    with st.sidebar.expander(label = f"Diversity Image {i+1} Settings"):
        
        # upload image
        img_file_buffer = st.file_uploader(f"Upload Diversity Image {i+1}")
        
        if img_file_buffer is not None:
            
            image = Image.open(img_file_buffer)            
            imgs[i] = np.array(image)
            st.write(f"Image {i+1} uploaded successfully!")
        
        # use image yes/no
        use_im[i] = st.checkbox(label = "Use Image in Estimation?", key = f"checkbox{i}", value = True)
        
        # specify diversity phases
        if div_type == "Defocus":
            defocuses[i] = st.number_input(label = f"Defocus Amount ({units})", value = 0.0, key = f"defocus{i}")

        if div_type == "Zernike Polynomials":
            num_phases[i] = int(st.number_input(label = "# of diversity phases", min_value = 1, step = 1, value = 1, key = f"num_phases{i}"))

            zern_idx[i] = [None for i in range(num_phases[i])]
            zern_str[i] = [None for i in range(num_phases[i])]

            for j in range(num_phases[i]):
                zern_idx[i][j] = st.number_input(label = f"Diversity Phase {j+1} Polynomial Index ({indexing})", min_value = 4, value = 4, step = 1, key = f"diversity_index{i},{j}")
                n_temp, m_temp = zn2nm(zern_idx[i][j], -1)
                st.write(f"n = {n_temp}, m = {m_temp}")
                zern_str[i][j] = st.number_input(label = f"Diversity Phase {j+1} Amount ({units})", value  = 0.0, key = f"diversity{i},{j}")
        
estimate_button = st.button("Run Estimation")

pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase
def_con = {"um":1.0, "waves":l, "radians": 1.0/l2p}
zern_con = {"um":l2p, "waves":2*np.pi, "radians":1.0}

if estimate_button:
    with st.spinner("Calculating ..."):
        
        im_inds = [i for i in range(len(use_im)) if use_im[i] == True]
        imgs_temp = np.array([imgs[i] for i in im_inds])
    
        imsize = imgs[0].shape[0]
        ff = Fast_FFTs(imsize, len(im_inds), 1)
    
        zern, R, Theta, inds = get_zern(imsize, pupilSize, pixelSize, num_c)
        
        c0 = np.zeros((num_c))+1e-10
        
        if div_type == "Defocus":
            def_vals = def_con[units]*np.array([defocuses[i] for i in im_inds])
            theta1 = defocus(def_vals, R, inds, NA, l, RI)
            
        if div_type == "Zernike Polynomials":
            
            j_max = max([max(zern_idx[i]) for i in range(num_imgs)])
            theta0 = np.zeros((num_imgs, j_max))
    
            for i in range(num_imgs):
                for j in range(num_phases[i]):
                    theta0[i][zern_idx[i][j]] = zern_str[i][j]
    
            get_theta(theta0, zern)
    
        if algo == "Poisson":
            c = iter_p(zern, inds, imgs_temp, theta1, Sk0, c0.copy(), ff)[0]
    
        if algo == "Gaussian":
            c = iter_g0(zern, inds, imgs_temp, theta1, Sk0, c0.copy(), ff, 100)[0]
            
        st.session_state.cs.append(c)

df = pd.DataFrame(data = st.session_state.cs, columns = [[f"{j}" for j in range(num_c)],
                                                         [f"{zn2nm(j, 3)[1]}, {zn2nm(j, 3)[0]}" for j in range(num_c)]])

with st.expander("Previous Results"):
    st.table(df.style.format("{:.2f}"))
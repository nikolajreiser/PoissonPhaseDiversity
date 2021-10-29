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
from src.sim_cell import cell_multi
from src.streamlit_functions import upload_image, run_estimation
st.title('Phase Diversity')

MAX_BUTTONS = 10

NA = st.sidebar.number_input(label = "NA", min_value = 0.0, value  = 1.2, step = 0.1)
pixelSize = st.sidebar.number_input(label = "Pixel Size (um)", format = "%.3f", min_value = 0.0, value  = .096)
l = st.sidebar.number_input(label = "Wavelength (um)", format = "%.3f", min_value = 0.0, value  = .532)
RI = st.sidebar.number_input(label = "Refractive Index", min_value = 0.0, value  = 1.33)
rotang = st.sidebar.number_input(label = "Pupil Angle Offset (degrees)", value  = 0.0)
num_c = st.sidebar.number_input(label = "# of orders to estimate", min_value = 12, step = 1)
num_imgs = st.sidebar.number_input(label = "# of images", min_value = 2, step = 1, value = 3)

pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase

img_buttons = {i: st.sidebar.button(f"Upload Image {i+1}") for i in range(num_imgs)}

estimate_button = st.button("Go", on_click = run_estimation)

for i, ib in img_buttons.items():
    if ib: st.session_state[f"imgs{i}"] = upload_image(i)
st.write(st.session_state)

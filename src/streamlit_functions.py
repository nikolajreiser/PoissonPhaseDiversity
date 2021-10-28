#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:08:11 2021

@author: nikolaj
"""

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from time import sleep

def upload_image(i):
        
    img_file_buffer = st.file_uploader(f"Upload Diversity Image {i+1}")
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer.data)
        img = np.array(image)
        st.write(f"Image {i+1} uploaded successfuly!")
        
        return img
    return -1

            
def run_estimation():
    imgs = [img for key, img in st.session_state.items() if 'img' in key]
    print("running estimation")
    for key, dat in st.session_state.items():
        print(key)

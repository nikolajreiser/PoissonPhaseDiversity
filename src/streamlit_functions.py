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

     
def run_estimation():
    imgs = [img for key, img in st.session_state.items() if 'img' in key]
    print("running estimation")
    for key, dat in st.session_state.items():
        print(key)

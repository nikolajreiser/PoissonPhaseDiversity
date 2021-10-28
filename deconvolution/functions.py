#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:16:32 2021

@author: nikolaj
"""

import numpy as np
import scipy

def sft(im): return np.fft.fftshift(im, axes = (-1, -2))
def fft(im): return scipy.fft.fft2(im)
def ift(im): return scipy.fft.ifft2(im)
def fft2(im): return scipy.fft.rfft2(im)
def ift2(im): return scipy.fft.irfft2(im)
def scl(im): return np.divide(im-im.min(), im.max()-im.min())

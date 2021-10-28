#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:13:56 2020

@author: nikolaj
"""


import pyfftw
import os
import pathlib

class Fast_FFTs():
    def __init__(self, dsize, num_imgs, num_threads = 1):
    
        hsize = dsize//2 + 1
        flags=('FFTW_MEASURE', )
        nt = num_threads
        export_wisdom = True
        
        wisdom = self.read_wisdom(dsize, num_imgs, nt)
        
        if wisdom != -1:
            pyfftw.import_wisdom(wisdom)
            export_wisdom = False

        c2cf_in = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'complex128')        
        c2cf_out = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'complex128')
        self.c2cf = pyfftw.FFTW(c2cf_in, c2cf_out, axes = (1,2), flags = flags, threads = nt)

        c2cf1_in = pyfftw.empty_aligned((dsize, dsize), dtype = 'complex128')        
        c2cf1_out = pyfftw.empty_aligned((dsize, dsize), dtype = 'complex128')
        self.c2cf1 = pyfftw.FFTW(c2cf1_in, c2cf1_out, axes = (0,1), flags = flags, threads = nt)


        c2cb_in = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'complex128')        
        c2cb_out = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'complex128')
        self.c2cb = pyfftw.FFTW(c2cb_in, c2cb_out, axes = (1,2), flags = flags, threads = nt,
                          direction='FFTW_BACKWARD', normalise_idft = True)
        
        c2cb1_in = pyfftw.empty_aligned((dsize, dsize), dtype = 'complex128')        
        c2cb1_out = pyfftw.empty_aligned((dsize, dsize), dtype = 'complex128')
        self.c2cb1 = pyfftw.FFTW(c2cb1_in, c2cb1_out, axes = (0,1), flags = flags, threads = nt,
                          direction='FFTW_BACKWARD', normalise_idft = True)
        
        r2c_in = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'float64')
        r2c_out = pyfftw.empty_aligned((num_imgs, dsize, hsize), dtype = 'complex128')    
        self.r2c = pyfftw.FFTW(r2c_in, r2c_out, axes = (1,2), flags = flags, threads = nt)
        
        r2c1_in = pyfftw.empty_aligned((dsize, dsize), dtype = 'float64')
        r2c1_out = pyfftw.empty_aligned((dsize, hsize), dtype = 'complex128')    
        self.r2c1 = pyfftw.FFTW(r2c1_in, r2c1_out, axes = (0,1), flags = flags, threads = nt)
        
        c2r_out = pyfftw.empty_aligned((num_imgs, dsize, dsize), dtype = 'float64')
        c2r_in = pyfftw.empty_aligned((num_imgs, dsize, hsize), dtype = 'complex128')
        self.c2r = pyfftw.FFTW(c2r_in, c2r_out, axes = (1,2), flags = flags, threads = nt, 
                          direction='FFTW_BACKWARD', normalise_idft = True)
            
        c2r1_out = pyfftw.empty_aligned((dsize, dsize), dtype = 'float64')
        c2r1_in = pyfftw.empty_aligned((dsize, hsize), dtype = 'complex128')
        self.c2r1 = pyfftw.FFTW(c2r1_in, c2r1_out, axes = (0,1), flags = flags, threads = nt, 
                          direction='FFTW_BACKWARD', normalise_idft = True)
        
        if export_wisdom: self.write_wisdom(dsize, num_imgs, nt)
        
    def fft(self, im):
        nd = len(im.shape)
        if nd == 2:
            self.c2cf1.input_array[:] = im
            return self.c2cf1().copy()
        if nd == 3:
            self.c2cf.input_array[:] = im
            return self.c2cf().copy()

    def ift(self, im):
        
        nd = len(im.shape)
        if nd == 2:
            self.c2cb1.input_array[:] = im
            return self.c2cb1().copy()
        if nd == 3:
            self.c2cb.input_array[:] = im
            return self.c2cb().copy()

    
    def fft2(self, im):

        nd = len(im.shape)
        if nd == 2:
            self.r2c1.input_array[:] = im
            return self.r2c1().copy()
        if nd == 3:
            self.r2c.input_array[:] = im
            return self.r2c().copy()
    
    def ift2(self, im):
        nd = len(im.shape)
        if nd == 2:
            self.c2r1.input_array[:] = im
            return self.c2r1().copy()
        if nd == 3:
            self.c2r.input_array[:] = im
            return self.c2r().copy()




    def write_wisdom(self, dsize, num_imgs, num_threads):
        
        wisdom = pyfftw.export_wisdom()
        w = f'wisdom_{dsize}_{num_imgs}_{num_threads}'
        p = pathlib.Path('wisdom')/w
        p.mkdir(parents = True, exist_ok = True)
        
        for i, l in enumerate(wisdom):
            file = open(p/f'{i}.txt', 'wb')
            file.write(l)
            file.close()
                
    def read_wisdom(self, dsize, num_imgs, num_threads):
        
        w = f'wisdom_{dsize}_{num_imgs}_{num_threads}'
        p = pathlib.Path('wisdom')/w
        
        if os.path.isdir(p):
            
            wisdom = []
            files = os.listdir(p)
            files.sort()
            
            for f in files:
                
                file = open(p/f, 'rb')
                wisdom.append(file.read())
                file.close()
            
            return wisdom
            
        else:
            return -1
    
    
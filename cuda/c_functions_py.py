   #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:28:42 2020

@author: nikolaj
"""

#compile command:
#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o c_functions.so --shared c_functions.cu -lcuda -lcublas -lcufft

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class CuFunc:
    
    def __init__(self, num_imgs, dsize, num_c):
        
        self.num_imgs = num_imgs
        self.dsize = dsize
        self.hsize = dsize//2 + 1
        self.dsize2d = dsize*dsize
        self.hsize2d = dsize*self.hsize
        self.dsize3d = num_imgs*self.dsize2d
        self.hsize3d = num_imgs*self.hsize2d
        
        lib = ctypes.cdll.LoadLibrary('./c_functions.so')
        self.lib = lib
        
        self.c_init_array = lib.init_array_host
        self.c_init_array.restype = ctypes.c_void_p
        self.c_init_array.argtpes = [ctypes.c_int]*2
        
        self.c_load_array = lib.load_array_host
        self.c_load_array.restype = ctypes.c_void_p
        self.c_load_array.argtypes = [ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        
        self.cu_load_array = lib.load_array
        self.cu_load_array.restype = ctypes.c_void_p
        self.cu_load_array.argtypes = [ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        
        self.cu_load_array_bool = lib.load_array_bool
        self.cu_load_array_bool.restype = ctypes.c_void_p
        self.cu_load_array_bool.argtypes = [ctypes.c_int, ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS")]
        
        self.cu_load_array_complex = lib.load_array_complex
        self.cu_load_array_complex.restype = ctypes.c_void_p
        self.cu_load_array_complex.argtypes = [ctypes.c_int, ndpointer(np.complex64, flags="C_CONTIGUOUS")]

        self.cu_load_array_int = lib.load_array_int
        self.cu_load_array_int.restype = ctypes.c_void_p
        self.cu_load_array_int.argtypes = [ctypes.c_int, ndpointer(np.int32, flags="C_CONTIGUOUS")]

        self.cu_unload_array = lib.unload_array
        self.cu_unload_array.restype = ndpointer(np.float32, shape = ((2416)), flags="C_CONTIGUOUS")
        self.cu_unload_array.argtypes = [ctypes.c_int, ctypes.c_void_p]
        
        self.cu_unload_array2 = lib.unload_array2
        self.cu_unload_array2.restype = ndpointer(np.float32, shape = ((num_imgs, dsize,dsize)), flags="C_CONTIGUOUS")
        self.cu_unload_array2.argtypes = [ctypes.c_int, ctypes.c_void_p]
        
        self.cu_unload_array3 = lib.unload_array2
        self.cu_unload_array3.restype = ndpointer(np.float32, shape = ((num_c)), flags="C_CONTIGUOUS")
        self.cu_unload_array3.argtypes = [ctypes.c_int, ctypes.c_void_p]

        
        self.cu_unload_array_complex = lib.unload_array_complex
        self.cu_unload_array_complex.restype = ndpointer(np.complex64, shape = ((num_imgs,dsize,dsize)), flags="C_CONTIGUOUS")
        self.cu_unload_array_complex.argtypes = [ctypes.c_int, ctypes.c_void_p]
        
        self.cu_unload_array_complex2 = lib.unload_array_complex2
        self.cu_unload_array_complex2.restype = ndpointer(np.complex64, shape = ((num_imgs,dsize,1+dsize//2)), flags="C_CONTIGUOUS")
        self.cu_unload_array_complex2.argtypes = [ctypes.c_int, ctypes.c_void_p]
        
        self.cu_unload_array_complex3 = lib.unload_array_complex3
        self.cu_unload_array_complex3.restype = ndpointer(np.complex64, shape = ((dsize,1+dsize//2)), flags="C_CONTIGUOUS")
        self.cu_unload_array_complex3.argtypes = [ctypes.c_int, ctypes.c_void_p]
        
        self.cu_init_array = lib.init_array
        self.cu_init_array.restype = ctypes.c_void_p
        self.cu_init_array.argtypes = [ctypes.c_int]*2
        
        self.cu_free_array = lib.free_array
        self.cu_free_array.restype = None
        self.cu_free_array.argtypes = [ctypes.c_void_p]
        
        self.cu_free_plan = lib.free_plan
        self.cu_free_plan.restype = None
        self.cu_free_plan.argtypes = [ctypes.c_void_p]
                
        self.cu_init_c2c = lib.init_c2c
        self.cu_init_c2c.restype = ctypes.c_void_p
        self.cu_init_c2c.argtypes = [ctypes.c_int]*2
        
        self.cu_ift_c2c = lib.ift_c2c
        self.cu_ift_c2c.restype = None
        self.cu_ift_c2c.argtypes = [ctypes.c_void_p]*3
        
        self.cu_abs2 = lib.abs2
        self.cu_abs2.restype = None
        self.cu_abs2.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*2
        
        self.cu_init_r2c = lib.init_r2c
        self.cu_init_r2c.restype = ctypes.c_void_p
        self.cu_init_r2c.argtypes = [ctypes.c_int]*2
        
        self.cu_fft_r2c = lib.fft_r2c
        self.cu_fft_r2c.restype = None
        self.cu_fft_r2c.argtypes = [ctypes.c_void_p]*3
        
        self.cu_init_r2cs = lib.init_r2c_single
        self.cu_init_r2cs.restype = ctypes.c_void_p
        self.cu_init_r2cs.argtypes = [ctypes.c_int]
        
        self.cu_init_c2rs = lib.init_c2r_single
        self.cu_init_c2rs.restype = ctypes.c_void_p
        self.cu_init_c2rs.argtypes = [ctypes.c_int]

        
        self.cu_mult1 = lib.mult1
        self.cu_mult1.restype = None
        self.cu_mult1.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*3
        
        self.cu_init_c2r = lib.init_c2r
        self.cu_init_c2r.restype = ctypes.c_void_p
        self.cu_init_c2r.argtypes = [ctypes.c_int]*2
        
        self.cu_ift_c2r = lib.ift_c2r
        self.cu_ift_c2r.restype = None
        self.cu_ift_c2r.argtypes = [ctypes.c_void_p]*3
        
        self.cu_quotient = lib.quotient
        self.cu_quotient.restype = None
        self.cu_quotient.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*3
        
        self.cu_cj = lib.cj
        self.cu_cj.restype = None
        self.cu_cj.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*2
        
        self.cu_mult_sum = lib.mult_sum
        self.cu_mult_sum.restype = None
        self.cu_mult_sum.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*3
        
        self.cu_mult_c = lib.mult_c
        self.cu_mult_c.restype = None
        self.cu_mult_c.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*3
        
        # self.cu_mult_r = lib.mult_r
        # self.cu_mult_r.restype = None
        # self.cu_mult_r.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*3
        
        
        self.cu_sim_im = lib.sim_im
        self.cu_sim_im.restype = ndpointer(np.float32, shape = ((num_imgs,dsize,dsize)), flags="C_CONTIGUOUS")
        self.cu_sim_im.argtypes = [ctypes.c_int]*4 + [ctypes.c_void_p]*17
        
        self.cu_init_cublas = lib.init_cublas
        self.cu_init_cublas.restype = ctypes.c_void_p
        self.cu_init_cublas.argtypes = None

        self.cu_ipc = lib.update_ob_poisson_cuda
        self.cu_ipc.restype = None
        self.cu_ipc.argtypes = [ctypes.c_int]*5 + [ctypes.c_void_p]*30 + [ctypes.c_float]


    def load_array(self, arr):
        
        size = 1
        dim = arr.shape
        for val in dim: size *= val
        
        if arr.dtype is np.dtype(np.float32):
            return self.cu_load_array(size, arr.flatten())
        if arr.dtype is np.dtype(np.complex64):
            return self.cu_load_array_complex(size, arr.flatten())
        if arr.dtype is np.dtype(np.bool):
            return self.cu_load_array_bool(size, arr.flatten())
        if arr.dtype is np.dtype(np.int32):
            return self.cu_load_array_int(size, arr.flatten())
        else:
            print("DTYPE NOT SUPPORTED")
            
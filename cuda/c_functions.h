#ifndef _C_FUNCTIONS_H_
#define _C_FUNCTIONS_H_

#include "c_functions.h"
#include "stdio.h"
#include "stdbool.h"
#include "cublas_v2.h"
#include "cuComplex.h"
#include "cufft.h"
#include "time.h"
#include "math.h"

#define DTYPE_R float
#define DTYPE_C cuFloatComplex
#define NUM_THREADS 256
#define NUM_MP 20

extern "C" void* init_array_host(int length, int dtype);
extern "C" void* load_array_host(int length, DTYPE_R* arr);
extern "C" DTYPE_R* load_array(int length, DTYPE_R* h_array);
extern "C" bool* load_array_bool(int length, bool* h_array);
extern "C" DTYPE_C* load_array_complex(int length, DTYPE_C* h_array);
extern "C" int* load_array_int(int length, int* h_array);
extern "C" DTYPE_R* unload_array(int length, DTYPE_R* d_array);
extern "C" DTYPE_R* unload_array2(int length, DTYPE_R* d_array);
extern "C" DTYPE_C* unload_array_complex(int length, DTYPE_C* d_array);
extern "C" DTYPE_C* unload_array_complex2(int length, DTYPE_C* d_array);
extern "C" DTYPE_C* unload_array_complex3(int length, DTYPE_C* d_array);
extern "C" void update_array(int length, DTYPE_R* d_array, DTYPE_R * h_array);
extern "C" void* init_array(int length, int dtype);
extern "C" void free_array(void* d_array);
extern "C" void free_plan(cufftHandle* plan);
extern "C" cufftHandle* init_c2c(int num_imgs, int length);
extern "C" void ift_c2c(cufftHandle* plan, DTYPE_C* H, DTYPE_C* h);
extern "C" void abs2(int length, DTYPE_C* h, DTYPE_R* s);
extern "C" cufftHandle* init_r2c(int num_imgs, int length);
extern "C" void fft_r2c(cufftHandle* plan, DTYPE_R* in, DTYPE_C* out);
extern "C" cufftHandle* init_r2c_single(int length);
extern "C" cufftHandle* init_c2r_single(int length);
extern "C" void mult1(int num_imgs, int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out);
extern "C" cufftHandle* init_c2r(int num_imgs, int length);
extern "C" void ift_c2r(cufftHandle* plan, DTYPE_C* in, DTYPE_R* out);
extern "C" void quotient(int length, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out);
extern "C" void cj(int length, DTYPE_C* in1, DTYPE_C* out);
extern "C" void mult_sum(int num_imgs, int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out);
extern "C" void mult_c(int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out);
extern "C" void mult_r(int length, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out);
extern "C" void update_ob_poisson_cuda(int num_imgs, int length, int num_c, int num_inds, int num_updates,
                DTYPE_C* F, DTYPE_C* theta, DTYPE_R* c, DTYPE_R* zern, 
                DTYPE_R* wavefront, DTYPE_C* H, DTYPE_C* h,
                DTYPE_R* s, DTYPE_C* S, DTYPE_C* G, DTYPE_R* g, 
                cufftHandle* r2c, cufftHandle* c2r, cufftHandle* c2c, cufftHandle* r2cs, cufftHandle* c2rs,
                cublasHandle_t* handle, int* inds, DTYPE_R* h_g,
                DTYPE_R* dU, DTYPE_C* dF, DTYPE_C* Q, DTYPE_R*f, DTYPE_R* df,
                DTYPE_R* q, DTYPE_R* inner, DTYPE_C* inner1, DTYPE_R* dL_integrand,
                DTYPE_R* dc, DTYPE_R* imgs, DTYPE_R nSk0);
extern "C" cublasHandle_t* init_cublas();
extern "C" DTYPE_R* sim_im(int num_imgs, int length, int num_c, int num_inds,
                DTYPE_C* F, DTYPE_C* theta, DTYPE_R* phi, DTYPE_R* zern, 
                DTYPE_R* wavefront, DTYPE_C* H, DTYPE_C* h,
                DTYPE_R* s, DTYPE_C* S, DTYPE_C* G, DTYPE_R* g, 
                cufftHandle* r2c, cufftHandle* c2r, cufftHandle* c2c,
                cublasHandle_t* handle,
                int* inds, DTYPE_R* h_g);

#endif


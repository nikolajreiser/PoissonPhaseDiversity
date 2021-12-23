#include "c_functions.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
#endif


void* load_array_host(int length, DTYPE_R* arr){
    void* h_arr;
    int size = length*sizeof(DTYPE_R);
    h_arr = malloc(size);
    return h_arr;
}

DTYPE_R* load_array(int length, DTYPE_R* h_array){


    DTYPE_R * d_array;
    int size = length*sizeof(DTYPE_R);

    gpuErrchk( cudaMallocManaged((DTYPE_R**)&d_array, size));
    gpuErrchk( cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    return d_array;    
}

bool* load_array_bool(int length, bool* h_array){


    bool * d_array;
    int size = length*sizeof(bool);

    gpuErrchk( cudaMallocManaged((DTYPE_R**)&d_array, size));
    gpuErrchk( cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    return d_array;    
}

DTYPE_C* load_array_complex(int length, DTYPE_C* h_array){


    DTYPE_C* d_array;
    int size = length*sizeof(DTYPE_C);

    gpuErrchk( cudaMallocManaged((DTYPE_C**)&d_array, size));
    gpuErrchk( cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    return d_array;    
}

int* load_array_int(int length, int* h_array){


    int* d_array;
    int size = length*sizeof(int);

    gpuErrchk( cudaMallocManaged((int**)&d_array, size));
    gpuErrchk( cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    return d_array;    
}


DTYPE_R* unload_array(int length, DTYPE_R* d_array){

    int size = length*sizeof(DTYPE_R);
    DTYPE_R* h_array = (DTYPE_R *) malloc(size);
    gpuErrchk( cudaMemcpy(h_array, (DTYPE_R *) d_array, size, cudaMemcpyDeviceToHost));

    return h_array;
}

DTYPE_R* unload_array2(int length, DTYPE_R* d_array){

    int size = length*sizeof(DTYPE_R);
    DTYPE_R* h_array = (DTYPE_R *) malloc(size);
    gpuErrchk( cudaMemcpy(h_array, (DTYPE_R *) d_array, size, cudaMemcpyDeviceToHost));

    return h_array;
}


DTYPE_C* unload_array_complex(int length, DTYPE_C* d_array){

    int size = length*sizeof(DTYPE_C);
    DTYPE_C* h_array = (DTYPE_C *) malloc(size);
    gpuErrchk( cudaMemcpy(h_array, (DTYPE_C *) d_array, size, cudaMemcpyDeviceToHost));

    return h_array;
}

DTYPE_C* unload_array_complex2(int length, DTYPE_C* d_array){

    int size = length*sizeof(DTYPE_C);
    DTYPE_C* h_array = (DTYPE_C *) malloc(size);
    gpuErrchk( cudaMemcpy(h_array, (DTYPE_C *) d_array, size, cudaMemcpyDeviceToHost));

    return h_array;
}

DTYPE_C* unload_array_complex3(int length, DTYPE_C* d_array){

    int size = length*sizeof(DTYPE_C);
    DTYPE_C* h_array = (DTYPE_C *) malloc(size);
    gpuErrchk( cudaMemcpy(h_array, (DTYPE_C *) d_array, size, cudaMemcpyDeviceToHost));

    return h_array;
}

void update_array(int length, DTYPE_R* d_array, DTYPE_R * h_array){
    int size = length*sizeof(DTYPE_R);
    gpuErrchk( cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));
}

void* init_array(int length, int dtype){

    void* d_array;
    if(dtype == 0){
        int size = length*sizeof(DTYPE_R);
        gpuErrchk( cudaMallocManaged((DTYPE_R**)&d_array, size));
        gpuErrchk( cudaMemset(d_array, 0, size));
    }
    
    else{
        int size = length*sizeof(DTYPE_C);
        gpuErrchk( cudaMallocManaged((DTYPE_C**)&d_array, size));
        gpuErrchk( cudaMemset(d_array, 0, size));

    }

    return d_array;
}

void* init_array_host(int length, int dtype){

    void* d_array;
    int size = 0;
    if(dtype == 0){
        size = length*sizeof(DTYPE_R);
    }
    
    else{
        size = length*sizeof(DTYPE_C);
    }
    
    d_array = malloc(size);
    return d_array;
}
void free_array(void* d_array){
    gpuErrchk( cudaFree(d_array));
}

void free_plan(cufftHandle* plan){
    cufftDestroy(*plan);
    free(plan);
}

void get_wavefront(cublasHandle_t* handle, int num_c, int length, 
                DTYPE_R* phi, DTYPE_R* zern, DTYPE_R* wavefront){
        
        DTYPE_R alpha = 1;
        DTYPE_R beta = 0;

    cublasSgemv(*handle, CUBLAS_OP_N,
                           length, num_c,
                           &alpha,
                           zern, length,
                           phi, 1,
                           &beta,
                           wavefront, 1);    
}


__global__  void wavefront2H_kernel(int l, int num_imgs, int l2d, int* inds, DTYPE_R* wavefront, DTYPE_C* theta, DTYPE_C* H){

    
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){

        DTYPE_R x_ang, y_ang;
        sincosf(wavefront[i], &y_ang, &x_ang);

        #pragma unroll
        for(int j = 0; j<num_imgs; j++){

            int t_idx = i+j*l;
            int H_idx = inds[i]+j*l2d;
                        
            //needs to be updated to make_cuXComplex for different dtype
            H[H_idx] = cuCmulf(theta[t_idx], make_cuFloatComplex(x_ang, y_ang));
        }
    }
}

void wavefront2H(int num_inds, int num_imgs, int l2d, int* inds, DTYPE_R* wavefront, DTYPE_C* theta, DTYPE_C* H){

    wavefront2H_kernel<<<32*NUM_MP, NUM_THREADS>>>(num_inds, num_imgs, l2d, inds, wavefront, theta, H);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

cufftHandle* init_c2c(int num_imgs, int length){

    cufftHandle* plan = (cufftHandle*) malloc(sizeof(cufftHandle));
    
    int n[2] = {length, length};
    if (cufftPlanMany(plan, 2, n,
    				  NULL, 1, 0,
    				  NULL, 1, 0,
    				  CUFFT_C2C, num_imgs) != CUFFT_SUCCESS){
    	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
    	}
    	    	
    return plan;	
}


void ift_c2c(cufftHandle* plan, DTYPE_C* H, DTYPE_C* h){

    cufftExecC2C(*plan, H, h, CUFFT_INVERSE);

}

__global__  void abs2_kernel(int n, DTYPE_C* in, DTYPE_R* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<n; i += blockDim.x*gridDim.x){
        out[i] = in[i].x*in[i].x+in[i].y*in[i].y;
    }
}

void abs2(int length, DTYPE_C* in, DTYPE_R* out){

    abs2_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in, out);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

cufftHandle* init_r2c(int num_imgs, int length){
    cufftHandle* plan = (cufftHandle*) malloc(sizeof(cufftHandle));
    int n[2] = {length, length};
    if (cufftPlanMany(plan, 2, n,
    				  NULL, 1, 0,
    				  NULL, 1, 0,
    				  CUFFT_R2C, num_imgs) != CUFFT_SUCCESS){
    	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
    	}
    return plan;
}

void fft_r2c(cufftHandle* plan, DTYPE_R* in, DTYPE_C* out){
    cufftExecR2C(*plan, in, out);
}

cufftHandle* init_r2c_single(int length){
    cufftHandle* plan = (cufftHandle*) malloc(sizeof(cufftHandle));

    if (cufftPlan2d(plan, length, length, CUFFT_R2C) != CUFFT_SUCCESS){
    	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
    }
    return plan;
}

cufftHandle* init_c2r_single(int length){
    cufftHandle* plan = (cufftHandle*) malloc(sizeof(cufftHandle));

    if (cufftPlan2d(plan, length, length, CUFFT_C2R) != CUFFT_SUCCESS){
    	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
    }
    return plan;
}


__global__  void mult_kernel1(int n, int l, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){

    int N = n*l;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<N; i += blockDim.x*gridDim.x){
        
        out[i] = cuCmulf(in1[i%l], in2[i]);
    }
}


void mult1(int num_imgs, int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){
                
    mult_kernel1<<<32*NUM_MP, NUM_THREADS>>>(num_imgs, length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__ void log_kernel(int n, DTYPE_R* in, DTYPE_R* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<n; i += blockDim.x*gridDim.x){
        out[i] = log(in[i]);
    }

}


void logFunc(int n, DTYPE_R* in, DTYPE_R* out){
    log_kernel<<<32*NUM_MP, NUM_THREADS>>>(n, in, out);

}

cufftHandle* init_c2r(int num_imgs, int length){
    cufftHandle* plan = (cufftHandle*) malloc(sizeof(cufftHandle));
    int n[2] = {length, length};
    if (cufftPlanMany(plan, 2, n,
    				  NULL, 1, 0,
    				  NULL, 1, 0,
    				  CUFFT_C2R, num_imgs) != CUFFT_SUCCESS){
    	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
    	}
    return plan;
}

void ift_c2r(cufftHandle* plan, DTYPE_C* in, DTYPE_R* out){
    cufftExecC2R(*plan, in, out);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void quotient_kernel(int n, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<n; i += blockDim.x*gridDim.x){
            
        out[i] = __ddiv_rn(in1[i], in2[i]);
    }
}


void quotient(int length, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out){
                
    quotient_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void quotient_c_kernel(int n, DTYPE_R* in1, DTYPE_C* in2, DTYPE_C* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<n; i += blockDim.x*gridDim.x){
          
        DTYPE_C temp = make_cuFloatComplex(in1[i], 0);
        out[i] = cuCdivf(temp, in2[i]);  

    }
}


void quotient_c(int length, DTYPE_R* in1, DTYPE_C* in2, DTYPE_C* out){
                
    quotient_c_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void conj_kernel(int n, DTYPE_C* in, DTYPE_C* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<n; i += blockDim.x*gridDim.x){
            
        out[i] = cuConjf(in[i]);
    }
}


void cj(int length, DTYPE_C* in, DTYPE_C* out){
                
    conj_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void mult_sum_kernel(int n, int l, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){
    
        DTYPE_C temp = make_cuFloatComplex(0.0, 0.0);
        #pragma unroll
        for(int j = 0; j<n; j++){
            int idx = i+j*l;
            temp = cuCaddf(cuCmulf(in1[idx], in2[idx]), temp);
        }
        out[i] = temp;
    }
}


void mult_sum(int num_imgs, int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){

    mult_sum_kernel<<<32*NUM_MP, NUM_THREADS>>>(num_imgs, length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void mult_c_kernel(int l, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){
    
         out[i] = cuCmulf(in1[i], in2[i]);
    }
}


void mult_c(int length, DTYPE_C* in1, DTYPE_C* in2, DTYPE_C* out){

    mult_c_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void mult_r_kernel(int l, DTYPE_R denom, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){
    
         out[i] = in1[i]*in2[i]/denom;
    }
}


void mult_r(int length, DTYPE_R denom, DTYPE_R* in1, DTYPE_R* in2, DTYPE_R* out){

    mult_r_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, denom, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void mult_rconj_kernel(int l, DTYPE_R* in1, DTYPE_C* in2, DTYPE_C* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){
    
         out[i] = make_cuFloatComplex(in1[i]*in2[i].x, -1.0*in1[i]*in2[i].y);
    }
}


void mult_rconj(int length, DTYPE_R* in1, DTYPE_C* in2, DTYPE_C* out){

    mult_rconj_kernel<<<32*NUM_MP, NUM_THREADS>>>(length, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


__global__  void mult_sum_im_kernel(int num_inds, int n, int l, int* inds, DTYPE_C* in1, DTYPE_C* in2, DTYPE_R* out){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<num_inds; i += blockDim.x*gridDim.x){
    
        DTYPE_R temp = 0;
        
        #pragma unroll
        for(int j = 0; j<n; j++){
            int idx = inds[i]+j*l;
            temp += in1[idx].x*in2[idx].y+in1[idx].y*in2[idx].x;
        }
        out[i] = temp;
    }
}


void mult_sum_im(int num_inds, int num_imgs, int length, int* inds, DTYPE_C* in1, DTYPE_C* in2, DTYPE_R* out){

    mult_sum_im_kernel<<<32*NUM_MP, NUM_THREADS>>>(num_inds, num_imgs, length, inds, in1, in2, out);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

__global__  void init_val(int l, DTYPE_R val,  DTYPE_R* arr){

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<l; i += blockDim.x*gridDim.x){
         arr[i] = val;
    }
}

cublasHandle_t* init_cublas(){
    cublasHandle_t* handle = (cublasHandle_t*) malloc(sizeof(cublasHandle_t));
    cublasCreate(handle);
    return handle;
}

DTYPE_R* sim_im(int num_imgs, int length, int num_c, int num_inds,
                DTYPE_C* F, DTYPE_C* theta, DTYPE_R* phi, DTYPE_R* zern, 
                DTYPE_R* wavefront, DTYPE_C* H, DTYPE_C* h,
                DTYPE_R* s, DTYPE_C* S, DTYPE_C* G, DTYPE_R* g, 
                cufftHandle* r2c, cufftHandle* c2r, cufftHandle* c2c,
                cublasHandle_t* handle, int* inds, DTYPE_R* h_g){
                        
    int hlength = length/2 + 1;
    int l2d = length*length;
    int h2d = length*hlength;
    int l3d = l2d*num_imgs;    
    int R = sizeof(DTYPE_R);

    
    get_wavefront(handle, num_c, num_inds, phi, zern, wavefront);        
    wavefront2H(num_inds, num_imgs, l2d, inds, wavefront, theta, H);    
    cufftExecC2C(*c2c, H, h, CUFFT_INVERSE);
    abs2(l3d, h, s);
    cufftExecR2C(*r2c, s, S);
    mult1(num_imgs, h2d, F, S, G);
    cufftExecC2R(*c2r, G, g);

    gpuErrchk( cudaMemcpy(h_g, g, l3d*R, cudaMemcpyDeviceToHost));
    
    //DTYPE_R result = -1;
    //cublasDnrm2(*handle, l3d, s, 1, &result);
    //printf("check: %.2e\n", result);

    return h_g;
}

void update_ob_poisson_cuda(int num_imgs, int length, int num_c, int num_inds, int num_updates,
                DTYPE_C* F, DTYPE_C* theta, DTYPE_R* c, DTYPE_R* zern, 
                DTYPE_R* wavefront, DTYPE_C* H, DTYPE_C* h,
                DTYPE_R* s, DTYPE_C* S, DTYPE_C* G, DTYPE_R* g, 
                cufftHandle* r2c, cufftHandle* c2r, cufftHandle* c2c, cufftHandle* r2cs, cufftHandle* c2rs,
                cublasHandle_t* handle, int* inds, DTYPE_R* h_g,
                DTYPE_R* dU, DTYPE_C* dF, DTYPE_C* Q, DTYPE_R*f, DTYPE_R* df,
                DTYPE_R* q, DTYPE_R* inner, DTYPE_C* inner1, DTYPE_R* dL_integrand,
                DTYPE_R* dc, DTYPE_R* imgs, DTYPE_R nSk0){
                

    int hlength = length/2 + 1;
    int l2d = length*length;
    int h2d = length*hlength;
    int l3d = l2d*num_imgs;
    int h3d = h2d*num_imgs;
    
    DTYPE_R* glog = (DTYPE_R *) init_array(l3d, 0);
    DTYPE_R* c_temp = (DTYPE_R *) init_array(num_c, 0);

    DTYPE_R coeff_scaling = 1/((double)l2d);
    DTYPE_R ift_scaling = coeff_scaling*coeff_scaling;

    int max_iter_linesearch = 10;
    DTYPE_R ss = 300;
    DTYPE_R ss_reduce = .3;


    clock_t start = clock(), diff;

    int max_iter = 1000;
    int min_iter = 200;
    DTYPE_R eps = 1e-3;
    DTYPE_R norm_g = 1+eps;
    int n_iter = 0;
    
    DTYPE_R L0 = -100000000000000000;
    DTYPE_R L1 = 0;
    
    
    while(true){

        //line search
        
        for(int i = 0; i < max_iter_linesearch; i++){

            for(int j = 0; j<num_c; j++){c_temp[j] = c[j] - ss*dc[j];}
            //compute wavefront (H)
            get_wavefront(handle, num_c, num_inds, c_temp, zern, wavefront);        
            wavefront2H(num_inds, num_imgs, l2d, inds, wavefront, theta, H); 
            
            //compute psf (s)  
            cufftExecC2C(*c2c, H, h, CUFFT_INVERSE);
            cublasCsscal(*handle, l3d, &ift_scaling, h, 1);
            
            abs2(l3d, h, s);
            cufftExecR2C(*r2c, s, S);
                

            //compute g
            cufftExecR2C(*r2cs, f, F);
            mult1(num_imgs, h2d, F, S, G);
            cufftExecC2R(*c2r, G, g);
            

            
            //compute cost function
            logFunc(l3d, g, glog);
            cublasSdot(*handle, l3d, imgs, 1, glog, 1, &L1);
            
            //if cost function is increasing, step size is good,
            //and the line search can be exited
            if(L1 > L0){break;}
            
            //if cost function is decreasing, step size must be reduced
            else{ss *= ss_reduce;}
                
        }

        //update coefficients and current cost function value
        L0 = L1;
        for(int j = 0; j<num_c; j++){c[j] = c[j] - ss*dc[j];}
        
        
        //compute poisson object update
        quotient(l3d, imgs, g, q);
        cufftExecR2C(*r2c, q, Q);
        cj(h3d, S, S);
        mult_sum(num_imgs, h2d, Q, S, dF);
        
        

    
        //update object
        cufftExecC2R(*c2rs, dF, df);
        cublasSscal(*handle, l2d, &nSk0, df, 1);        
        mult_r(l2d, nSk0, df, f, f);    
    

    
    
        //get coefficient update vector
        cj(h2d, F, F);
        mult1(num_imgs, h2d, F, Q, Q);
        cufftExecC2R(*c2r, Q, inner);
        mult_rconj(l3d, inner, h, inner1);      
          
        

        cufftExecC2C(*c2c, inner1, inner1, CUFFT_INVERSE);
        cublasCsscal(*handle, l3d, &ift_scaling, inner1, 1);
        mult_sum_im(num_inds, num_imgs, l2d, inds, inner1, H, dL_integrand);
        
        //undo changes to object
        cj(h2d, F, F);
        

        //update coefficients
        for(int i = 0; i<num_c; i++){
            cublasSdot(*handle, num_inds, dL_integrand, 1, &(zern[i*num_inds]), 1, &(dc[i]));
            dc[i] *= 2*coeff_scaling;
            //printf("%.2e\n", dc[i]);
        }
    

    
        //stopping conditions
        if(n_iter>=max_iter){break;}
    
        if(n_iter>min_iter){
            
            //terminate if total step size is small
            norm_g = 0;
            for(int j = 0; j<num_c; j++){norm_g += dc[j]*dc[j];}
            if(sqrt(norm_g)*ss < eps){break;}
        }
        n_iter += 1;


    }
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        printf("%i iterations\n", n_iter);


}
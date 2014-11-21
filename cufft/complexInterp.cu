#include "luaT.h"
#include "THC/THC.h"
#include "arithmetic.h"
#include "cufft.h"

// Both interpolation kernels and inputs are copied into shared memory. 
// Each thread compute one element of the output. 
__global__ void batch_interpolate_kernel_forward(cuComplex* input, cuComplex* output, 
                                         float* kernelRows, float* kernelCols,
                                         const int iH, const int iW,
                                         const int oH, const int oW,
                                         const int nPlanes){
  extern __shared__ float shared_mem[];
  float* R = (float*)shared_mem;
  float* C = (float*)&R[iH*oH];
  cuComplex* S = (cuComplex*)&C[iW*oW];

  const int plane = blockIdx.x;
  if (plane >= nPlanes)
    return;

  input += plane * iH * iW;
  output += plane * oH * oW;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // copy the interpolation kernels and input to shared memory  
  if (ty < iH) { 
    R[ty*oH + tx] = kernelRows[ty*oH + tx];
    C[ty*oW + tx] = kernelCols[ty*oW + tx];
  }
  __syncthreads();

  if (tx <  iW && ty < iH)
    S[ty*iW + tx] = input[ty*iW + tx];
  __syncthreads();
  
  // compute result
  float real = 0;
  float imag = 0;
 
  for (int i = 0; i < iH; ++i) {
    for (int j = 0; j < iW; ++j) {
      real += S[i*iW + j].x * R[i*oH + ty] * C[j*oW + tx];
      imag += S[i*iW + j].y * R[i*oH + ty] * C[j*oW + tx];
    }
  }
  output[ty*oW + tx].x = real;
  output[ty*oW + tx].y = imag;
  __syncthreads();
}
    

// Similar to above kernel, but in the case that the output is smaller than the input. 
__global__ void batch_interpolate_kernel_backward(cuComplex* input, cuComplex* output, 
                                         float* kernelRows, float* kernelCols,
                                         const int iH, const int iW,
                                         const int oH, const int oW,
                                         const int nPlanes){
  extern __shared__ float shared_mem[];
  float* R = (float*)&shared_mem;
  float* C = (float*)&R[iH*oH];
  cuComplex* S = (cuComplex*)&C[iW*oW];

  const int plane = blockIdx.x;
  if (plane >= nPlanes)
    return;
  const int N = iH;
  const int K = oH;
  input += plane * N * N;
  output += plane * K * K;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // copy the interpolation kernels and input to shared memory  
  if (ty < oH) { 
    R[ty*N + tx] = kernelRows[ty*N + tx];
    C[ty*N + tx] = kernelCols[ty*N + tx];
  }
  __syncthreads();

  if (tx <  N && ty < N) {
    S[ty*N + tx] = input[ty*N + tx];
  }
  __syncthreads();
  
  if (tx < K && ty < K) {
    // compute
    float real = 0;
    float imag = 0;
 
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        real += S[j*N + i].x * R[ty*N + j] * C[tx*N + i];
        imag += S[j*N + i].y * R[ty*N + j] * C[tx*N + i];
      }
    }
    __syncthreads();
    output[ty*K + tx].x = real;
    output[ty*K + tx].y = imag;
  }
  __syncthreads();
}


static int complexInterp_interpolate(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *kernelRows = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *kernelCols = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  
  const int dim = input->nDimension;
  const long iH = input->size[dim-3];
  const long iW = input->size[dim-2];
  const long oH = output->size[dim-3];
  const long oW = output->size[dim-2];
  long nPlanes, nInputPlanes, nOutputPlanes;
  bool resize = false;

  luaL_argcheck(L, input->nDimension == output->nDimension, 2, "dimension mismatch");
  luaL_argcheck(L, input->size[0] == output->size[0], 2,"planes mismatch");

  if (dim == 5) {
    resize = true;
    luaL_argcheck(L, input->size[1] == output->size[1], 2,"planes mismatch");
    nOutputPlanes = input->size[0];
    nInputPlanes = input->size[1];
    nPlanes = nInputPlanes*nOutputPlanes;
    THCudaTensor_resize4d(input, nPlanes, iH, iW, 2);
    THCudaTensor_resize4d(output, nPlanes, oH, oW, 2);
  }
  else {
    nPlanes = input->size[0];
  }

  cuComplex* input_data = (cuComplex*)THCudaTensor_data(input);
  cuComplex* output_data = (cuComplex*)THCudaTensor_data(output);
  float* kernelRows_data = THCudaTensor_data(kernelRows);
  float* kernelCols_data = THCudaTensor_data(kernelCols);
  
  assert(iH == iW);
  assert(oH == oW);
  if (oH > iH && oW > iW) { 
    dim3 threads(oH,oW);
    dim3 blocks(nPlanes);
    int size = (iH*oH + iW*oW)*sizeof(float) + iH*iW*sizeof(cuComplex);
    batch_interpolate_kernel_forward<<<blocks,threads, size>>>(input_data, output_data, 
                                                               kernelRows_data, kernelCols_data,
                                                               iH, iW, oH, oW, nPlanes);
  }
  else if (oH <= iH && oW <= iW) {
    dim3 threads(iH,iW);
    dim3 blocks(nPlanes);
    int size = (iH*oH + iW*oW)*sizeof(float) + iH*iW*sizeof(cuComplex);
    batch_interpolate_kernel_backward<<<blocks,threads, size>>>(input_data, output_data, 
                                                                kernelRows_data, kernelCols_data,
                                                                iH, iW, oH, oW, nPlanes);
  }
  else {
    printf("error, WTF\n");
  }

  if (resize) {
    THCudaTensor_resize5d(input, nOutputPlanes, nInputPlanes, iH, iW, 2);
    THCudaTensor_resize5d(output, nOutputPlanes, nInputPlanes, oH, oW, 2);
  }

  CUDA_LOOK_FOR_ERROR();
  return 0;
}



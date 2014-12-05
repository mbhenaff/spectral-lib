#ifndef __SP_FFT_BIAS__
#define __SP_FFT_BIAS__

#include "THC.h"

// _add_bias and _fill_gradBias : code from torch (likely by Sixin)

// blockIdx.x  -> d
// threadIdx.x -> (m,n) [+blockDim.x]
// threadIdx.y -> z [+blockDim.y]
static __global__ void _add_bias(const float *bias, float *output,
				 int batch_n, int output_n, int output_h,
				 int output_w) {
  output += blockIdx.x*output_h*output_w;
  float b = bias[blockIdx.x];
  int oz,oxy;
  for (oz = threadIdx.y; oz < batch_n; oz += blockDim.y) {
    float *out = output + oz*output_n*output_h*output_w;
    for (oxy = threadIdx.x; oxy < output_h*output_w; oxy += blockDim.x) {
      out[oxy] += b;
    }
  }
}

// ASSUME
//  dim3 blocks(nOutputPlane);
//  dim3 threads(32,4);
// blockIdx.x  -> d
// threadIdx.x -> (m,n) [+blockDim.x]
// threadIdx.y -> z [+blockDim.y]
__global__ void _fill_gradBias(float *gradBias, const float *gradOutput, float scale,
			      int batch_n, int output_n,
			      int output_h, int output_w) {
  gradOutput += blockIdx.x*output_h*output_w;
  __shared__ float shGrad[128]; // 32*4
  float g = .0f;
  int oz,oxy;
  for (oz = threadIdx.y; oz < batch_n; oz += 4) {
    const float *out = gradOutput + oz*output_n*output_h*output_w;
    for (oxy = threadIdx.x; oxy < output_h*output_w; oxy += 32) {
      g += out[oxy];
    }
  }
  shGrad[threadIdx.y*blockDim.x+threadIdx.x] = g;
  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    g = .0f;
    for (oxy = 0; oxy < 128; ++oxy)
      g += shGrad[oxy];
    gradBias[blockIdx.x] = scale*g;
  }
}

#endif
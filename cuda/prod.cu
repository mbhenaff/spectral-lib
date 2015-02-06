#include "cuda_common.h"
#include <cassert>


__global__ void cuda_spectral_prod(const float* inputF, const float* kernelF, float* outputF, 
                                   const int dim, 
                                   const int M, const int ism, const int osm,
                                   const int F, const int isf, const int ksf,
                                   const int G, const int ksg, const int osg) {

  const int m0 = blockIdx.y;
  const int g0 = blockIdx.z;
  const int indx = blockIdx.x*blockDim.x + threadIdx.x;

  if (indx >= dim)
    return;  

  inputF  += m0 * ism           + indx;
  kernelF +=             g0*ksg + indx;
  outputF += m0 * osm  + g0*osg + indx;

  float val = 0;
  for (int f = 0; f < F; ++f, inputF += isf, kernelF +=ksf) {
    val += inputF[0]*kernelF[0];
  }
  outputF[0] = val;
}


void spectral_prod(const float* inputF, const float* kernelF, float* outputF, 
                   const int dim, 
                   const int M, const int ism, const int osm, 
                   const int F, const int isf, const int ksf, 
                   const int G, const int ksg, const int osg) {

  const int max_threads = 512;
  const int threadsPerBlock = min(dim,max_threads);
  const int blocksPerSample = max(DIVUP(dim, max_threads), 1);
  dim3 threads(threadsPerBlock,1);
  dim3 blocks(blocksPerSample, M, G);
  cuda_spectral_prod<<<blocks, threads>>>(inputF, kernelF, outputF, 
                                          dim, 
                                          M, ism, osm, 
                                          F, isf, ksf,
                                          G, ksg, osg);
  CUDA_LOOK_FOR_ERROR();
}
                                          




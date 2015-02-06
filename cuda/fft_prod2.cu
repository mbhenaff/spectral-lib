#include "fft_prod.h"
#include <cuComplex.h>
#include "cuda_common.h"
#include <cassert>

// o(m, g, y, x) = sum(f=0..F)sum(i=0..kH)sum(j=0..kW)
//                   i(m, f, y+i, x+j) k(g, f, i, j)
// m \in 0..(M-1), stride in input : ism, stride in output : osm
// f \in 0..(F-1), ...
// g \in 0..(G-1), ...
template<int nCache>
__global__ void cuda_fourier_prod(const cuComplex* inputF,
				  const cuComplex* kernelF,
				  cuComplex* outputF,
				  const int nRows, const int nCols,
				  const int M, const int ism, const int osm,
				  const int F, const int isf, const int ksf,
                  const int G, const int ksg, const int osg,const bool conj_kernel ) {
  //const int x  = threadIdx.x;
  //const int y  = blockIdx.x * blockDim.y + threadIdx.y;
  const int m0 = blockIdx.y * nCache;
  const int g0 = blockIdx.z * nCache;
  
  const int x = blockIdx.x * blockDim.y + threadIdx.y;
  const int y = threadIdx.x;


  inputF  += m0 * ism           + y*nCols + x;
  kernelF +=             g0*ksg + y*nCols + x;
  outputF += m0 * osm  + g0*osg + y*nCols + x;
  
  cuComplex  inputCache[nCache];
  cuComplex kernelCache[nCache];
  cuComplex outputCache[nCache*nCache];
  for (int i = 0; i < nCache*nCache; ++i)
    outputCache[i] = make_cuComplex(0.f, 0.f);
  
  for (int f = 0; f < F; ++f, inputF += isf, kernelF += ksf) {
    for (int a = 0; a < nCache; ++a) {
      inputCache [a] = inputF [a*ism];
      if(conj_kernel)
        kernelCache[a] = cuConjf(kernelF[a*ksg]);
      else
        kernelCache[a] = kernelF[a*ksg];
    }
    for (int m = 0; m < nCache; ++m)
      for (int g = 0; g < nCache; ++g)
	outputCache[m*nCache + g] =
	  cuCfmaf(inputCache[m], kernelCache[g], outputCache[m*nCache + g]);

  }

  for (int m = 0; m < nCache; ++m)
    for (int g = 0; g < nCache; ++g)
      outputF[m*osm + g*osg] = outputCache[m*nCache + g];
}


void fourier_prod(const cuComplex* inputF,
		  const cuComplex* kernelF,
		  cuComplex* outputF,
		  const int nRows, const int nCols,
		  const int M, const int ism, const int osm,
		  const int F, const int isf, const int ksf,
          const int G, const int ksg, const int osg, const bool conj_kernel) {
  const int nCache = 1;
  //assert(nRows % nCache == 0);
  //assert(nCols % nCache == 0);
  assert(M % nCache == 0);
  assert(G % nCache == 0);
  //assert(128 % nRows == 0);
  //assert(128 % nCols == 0); 
  const int nColPerBlock = min(nCols, max(128/nCols, 1));
  //printf("nRows=%d,nCols=%d,nColsPerBlock=%d\n",nRows,nCols,nColPerBlock);
  dim3 blocks(nCols/nColPerBlock, M/nCache, G/nCache);
  dim3 threads(nRows, nColPerBlock);
  cuda_fourier_prod<nCache><<<blocks, threads>>>(inputF, kernelF, outputF,
						 nRows,nCols, M, ism, osm, F, isf, ksf,
                         G, ksg, osg,conj_kernel);
  CUDA_LOOK_FOR_ERROR();
}

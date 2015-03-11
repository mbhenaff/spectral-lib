#include "common.hpp"
#include "fft_product2.hpp"
#include <iostream>
#include "arithmetic.hpp"
//#include "fft.hpp"
using namespace std;

template<bool accumulate> __device__
static inline void assignAcc(cuComplex & out, const cuComplex toAcc);
template<> __device__
inline void assignAcc<true>(cuComplex & out, const cuComplex toAcc) {
  out = cuCaddf(out, toAcc);
}
template<> __device__
inline void assignAcc<false>(cuComplex & out, const cuComplex toAcc) {
  out = toAcc;
}

template<bool conjugateKernel> __device__
static inline void assignKernelCache(cuComplex & out, const cuComplex in);
template<> __device__
inline void assignKernelCache<true>(cuComplex & out, const cuComplex in) {
  out = cuConjf(in);
}
template<> __device__
inline void assignKernelCache<false>(cuComplex & out, const cuComplex in) {
  out = in;
}

//TODO: this should be a matrix product using cublas
// o(m, g, y, x) = sum(f=0..F)sum(i=0..kH)sum(j=0..kW)
//                   i(m, f, y+i, x+j) k(g, f, i, j)
// outputF(m, g, y, x) = sum(f=0..F) inputF(m, f, y, x) kernelF(g, f, y, x)
// m \in 0..(M-1), stride in input : ism, stride in output : osm
// f \in 0..(F-1), ...
// g \in 0..(G-1), ...
template<int nCacheIn, int nCacheKer, bool accumulate, bool conjugateKernel> __global__
void fft_product_cudakernel(const cuComplex* inputF,
			    const cuComplex* kernelF,
			    cuComplex* outputF,
                const int nRows, const int nCols,
			    const int M, const int ism, const int osm,
			    const int F, const int isf, const int ksf,
                const int G, const int ksg, const int osg) {
  const int y  = blockIdx.x * blockDim.y + threadIdx.y;
  //if (y >= N/2+1)
  //  return;
  if (y >= nRows)
    return;

  const int x  = threadIdx.x;
  const int m0 = blockIdx.y * nCacheIn;
  const int g0 = blockIdx.z * nCacheKer;
  
  inputF  += m0 * ism           + y*nCols + x;
  kernelF +=             g0*ksg + y*nCols + x;
  outputF += m0 * osm  + g0*osg + y*nCols + x;

  /*
  inputF  += m0 * ism           + y*N + x;
  kernelF +=             g0*ksg + y*N + x;
  outputF += m0 * osm  + g0*osg + y*N + x;
  */
  cuComplex inputCache [nCacheIn];
  cuComplex kernelCache[nCacheKer];
  cuComplex outputCache[nCacheIn*nCacheKer];
  for (int i = 0; i < nCacheIn*nCacheKer; ++i)
    outputCache[i] = make_cuComplex(0.f, 0.f);
  
  for (int f = 0; f < F; ++f, inputF += isf, kernelF += ksf) {
    for (int a = 0; a < nCacheIn; ++a)
      inputCache [a] = inputF [a*ism];
    for (int a = 0; a < nCacheKer; ++a){
      assignKernelCache<conjugateKernel>(kernelCache[a], kernelF[a*ksg]);
    }
      /*
      if(conjugateKernel) 
        kernelCache[a] = cuConjf(kernelF[a*ksg]);
      else
        kernelCache[a] = kernelF[a*ksg];
      */
    for (int m = 0; m < nCacheIn; ++m)
      for (int g = 0; g < nCacheKer; ++g)
	outputCache[m*nCacheKer + g] =
	  cuCfmaf(inputCache[m], kernelCache[g], outputCache[m*nCacheKer + g]);

  }

  for (int m = 0; m < nCacheIn; ++m)
    for (int g = 0; g < nCacheKer; ++g)
      assignAcc<accumulate>(outputF[m*osm + g*osg], outputCache[m*nCacheKer + g]);
}

template<int nCacheIn, int nCacheKer>
void fft_product_nCaches(const cuComplex* inputF,
			 const cuComplex* kernelF,
			 cuComplex* outputF,
             const int nRows, const int nCols,
			 const int M, const int ism, const int osm,
			 const int F, const int isf, const int ksf,
			 const int G, const int ksg, const int osg,
             const bool accumulate, const bool conjugateKernel) {
  fft_assert(M % nCacheIn == 0);
  fft_assert(G % nCacheKer == 0);
  const int nLinesPerBlock = min(nRows, max(128/nCols,1));
  //const int nLinesPerBlock = min(N/2+1, max(128/N, 1)); // TODO: is 128 optimal ?
  //TODO: we could reuse the unused last y's in the next block
  //printf("grid size=%d x %d x %d\n",DIVUP(nRows,nLinesPerBlock), M/nCacheIn, G/nCacheKer);
  //printf("block size=%d x %d\n",nCols,nLinesPerBlock);
  dim3 blocks(DIVUP(nRows,nLinesPerBlock), M/nCacheIn, G/nCacheKer);
  dim3 threads(nCols, nLinesPerBlock);
  //dim3 blocks(DIVUP(N/2+1,nLinesPerBlock), M/nCacheIn, G/nCacheKer);
  //dim3 threads(N, nLinesPerBlock);
  if (accumulate){
    if (conjugateKernel)
      fft_product_cudakernel<nCacheIn, nCacheKer, true, true><<<blocks, threads>>>
        (inputF, kernelF, outputF, nRows, nCols, M, ism, osm, F, isf, ksf, G, ksg, osg);
    else
      fft_product_cudakernel<nCacheIn, nCacheKer, true, false><<<blocks, threads>>>
        (inputF, kernelF, outputF, nRows, nCols, M, ism, osm, F, isf, ksf, G, ksg, osg);
  }
  else {
    if (conjugateKernel)
      fft_product_cudakernel<nCacheIn, nCacheKer, false, true><<<blocks, threads>>>
        (inputF, kernelF, outputF, nRows, nCols, M, ism, osm, F, isf, ksf, G, ksg, osg);
    else
      fft_product_cudakernel<nCacheIn, nCacheKer, false, false><<<blocks, threads>>>
        (inputF, kernelF, outputF, nRows, nCols, M, ism, osm, F, isf, ksf, G, ksg, osg);
  }
  CUDA_LOOK_FOR_ERROR();
}

void fft_product_call(const cuComplex* inputF,
		      const cuComplex* kernelF,
		      cuComplex* outputF,
              const int nRows, const int nCols,
		      const int M, const int ism, const int osm,
		      const int F, const int isf, const int ksf,
		      const int G, const int ksg, const int osg,
              const bool accumulate, const bool conjugateKernel) {
  if (M % 4 == 0) {
    const int nCacheIn = 4;
    if (G % 4 == 0) {
      const int nCacheKer = 4;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    } else if (G % 3 == 0) {
      const int nCacheKer = 3;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate,conjugateKernel);
    } else {
      const int nCacheKer = 1;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    }
  } else if (M % 3 == 0) {
    const int nCacheIn = 3;
    if (G % 4 == 0) {
      const int nCacheKer = 4;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    } else if (G % 3 == 0) {
      const int nCacheKer = 3;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    } else {
      const int nCacheKer = 1;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    }
  } else {
    const int nCacheIn = 1;
    if (G % 4 == 0) {
      const int nCacheKer = 4;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    } else if (G % 3 == 0) {
      const int nCacheKer = 3;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    } else {
      const int nCacheKer = 1;
      fft_product_nCaches<nCacheIn, nCacheKer>(inputF, kernelF, outputF, nRows, nCols,
					       M, ism, osm, F, isf, ksf,
                           G, ksg, osg, accumulate, conjugateKernel);
    }
  }
}

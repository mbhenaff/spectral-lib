#include <cassert>
#include <algorithm>
#include "cuda_common.h"
#include "arithmetic.h"
#include "fft_cuda.h"
#include "fft_prod.h"
#include "extract_output.h"
#include "accGradParameters.h"
using namespace std;

// sizes:
//  input       : nMinibatches  * nInputPlanes  * iH * iW
//  gradOutput  : nMinibatches  * nOutputPlanes * oH * oW
//  gradWeight  : nOutputPlanes * nInputPlanes  * kH * kW
//  buffer      : linear memory with size at least (in bytes) :
//    sizeof(cuComplex)*N*N * max(nMinibatches * (nInputPlanes + nOutputPlanes) +
//                                nInputPlanes * nOutputPlanes,
//                                2 * nInputPlanes * nOutputPlanes)
// where:
//  oH  = iH-kH+1
//  oW  = iW-kW+1
//  lgN = ceil(log2(max(iH, iW)))
//  N   = 2^lgN
void cuda_fft_accGradParameters(const float* input,
				const float* gradOutput,
				float* gradWeight,
				cuComplex* buffer,
				const int lgN, const int nMinibatches,
				const int nInputPlanes, const int nOutputPlanes,
				const int iH, const int iW,
				const int kH, const int kW) {
  const int N = ipow(2, lgN);
  const int oH = iH - kH + 1;
  const int oW = iW - kW + 1;
  
  // tmp :
  cuComplex* gradWeightF = buffer;
  cuComplex* inputF = gradWeightF + N*N*nInputPlanes*nOutputPlanes;
  cuComplex* gradOutputF = inputF + N*N*nInputPlanes*nMinibatches;
  cuComplex* gradWeightT = inputF;

  cuda_fft<true, false>(input, inputF, nMinibatches*nInputPlanes, iH, iW, lgN);
  cuda_fft<true, true >(gradOutput, gradOutputF, nMinibatches*nOutputPlanes,
			oH, oW, lgN);

  fourier_prod(inputF, gradOutputF, gradWeightF, N,
	       nInputPlanes, N*N, N*N,
	       nMinibatches, nInputPlanes*N*N, nOutputPlanes*N*N,
	       nOutputPlanes, N*N, nInputPlanes*N*N);

  cuda_ffti<true, false>(gradWeightF, gradWeightT, nOutputPlanes*nInputPlanes, lgN);

  fft_extract_output(gradWeightT, gradWeight, nInputPlanes*nOutputPlanes,
		     N, iH, iW, kH, kW);

  CUDA_LOOK_FOR_ERROR();
}
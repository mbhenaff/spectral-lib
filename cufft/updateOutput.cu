#include <cassert>
#include <algorithm>
#include "cuda_common.h"
#include "arithmetic.h"
#include "fft_cuda.h"
#include "fft_prod.h"
#include "extract_output.h"
#include "updateOutput.h"
using namespace std;

void cuda_fft_updateOutput(const float* input,
			   const float* weight,
			   float* output,
			   cuComplex* buffer,
			   const int lgN, const int nMinibatches,
			   const int nInputPlanes, const int nOutputPlanes,
			   const int iW, const int kW) {
  const int N = ipow(2, lgN);
  const int oW = iW - kW + 1;

  // tmp
  cuComplex* outputF = buffer;
  cuComplex* inputF = outputF + N*nMinibatches*nOutputPlanes;
  cuComplex* weightF = inputF + N*nMinibatches*nInputPlanes;
  cuComplex* outputT = inputF;

  cuda_fft<true, false>(input, inputF, nMinibatches*nInputPlanes, iW, lgN);
  cuda_fft<true, true >(weight, weightF, nOutputPlanes*nInputPlanes, kW, lgN);
  

  fourier_prod(inputF, weightF, outputF, N,
	       nMinibatches, nInputPlanes*N*N, nOutputPlanes*N*N,
	       nInputPlanes, N*N, N*N,
	       nOutputPlanes, nInputPlanes*N*N, N*N);

  cuda_ffti<true, false>(outputF, outputT, nMinibatches*nOutputPlanes, lgN);
  
  fft_extract_output(outputT, output, nMinibatches*nOutputPlanes, N,
		     iH, iW, oH, oW);

  CUDA_LOOK_FOR_ERROR();
}

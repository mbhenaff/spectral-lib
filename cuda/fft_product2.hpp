// -*-cuda-*-

#include <cuComplex.h>
#include "common.hpp"
#include "cuda_common.hpp"

void fft_product_call(const cuComplex* inputF,
		      const cuComplex* kernelF,
		      cuComplex* outputF,
              const int nRows, const int nCols,
		      const int M, const int ism, const int osm,
		      const int F, const int isf, const int ksf,
		      const int G, const int ksg, const int osg,
		      const bool accumulate = false);

#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>

#define CUDA_CHECK(a)							\
  {									\
    cudaError_t status = (a);						\
    if (status != cudaSuccess) {					\
      std::cerr << "Error file " << __FILE__ <<				\
	" line " << __LINE__ << std::endl;				\
      std::cerr << cudaGetErrorString(status) << std::endl;		\
      exit(0);								\
    }									\
  }

#define CUDA_LOOK_FOR_ERROR()                                           \
  {                                                                     \
    cudaError_t err;							\
    if ((err = cudaGetLastError()) != cudaSuccess) {			\
      std::cerr << "Error in file " << __FILE__ <<			\
	" before line " << __LINE__ <<					\
	" : " << cudaGetErrorString(err) << std::endl;			\
      exit(0);								\
    }									\
  }


template<typename T>
T* cudaAllocCopy(T* data, size_t n) {
  T* output;
  CUDA_CHECK(cudaMalloc(&output, n*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(output, data, n*sizeof(T), cudaMemcpyHostToDevice));
  return output;
}


#endif

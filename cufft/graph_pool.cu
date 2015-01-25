#include "cuda_common.h"
#include <cassert>

__global__ void cuda_graph_pool_fprop(const float* input, const float* clusters, float* output, float* indices,
                                const int nMaps, const int dim, const int poolsize, const int nClusters, const int nClustersPerThread) {

  extern __shared__ float shared_mem[];
  float* input_data = (float*)shared_mem;
  float* cluster_indx = (float*)&input_data[dim];
  const int tidx = threadIdx.x;
  
  input += blockIdx.x * dim;
  output += blockIdx.x * nClusters;
  indices += blockIdx.x * nClusters;
  __syncthreads();
  // copy input data to shared memory
  int nChunks = DIVUP(dim, nClusters);
  int idx;
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nClusters;
    if (idx < dim)
      input_data[idx] = input[idx];
  }
  __syncthreads();
  
  // copy cluster indices to shared memory 
  const int nClusterIndices = nClusters * poolsize;
  nChunks = DIVUP(nClusterIndices,poolsize);
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nClusters;
    if (idx < nClusterIndices) 
      cluster_indx[idx] = clusters[idx]-1.0;
  }
  __syncthreads();
  
  /*
  for (int i = 0; i < nClusters; ++i) {
    const int idx = tidx + i*poolsize;
    if (idx < dim) {
      input_data[idx] = input[idx];
      cluster_indx[idx] = clusters[idx]-1.0;
    }
  }
  __syncthreads();
  */

  cluster_indx += threadIdx.x * nClustersPerThread * poolsize;
  output += threadIdx.x * nClustersPerThread;
  indices += threadIdx.x * nClustersPerThread;

  __syncthreads();
  float max; 
  int indx, indx_max;
  for (int i = 0; i < nClustersPerThread; ++i) {
    indx_max = (int)cluster_indx[i*poolsize];
    max = input_data[indx_max];
    for (int j = 1; j < poolsize; ++j) {
      indx = (int)cluster_indx[i*poolsize + j];
      if(input_data[indx] > max) {
        max = input_data[indx];
        indx_max = indx;
      } 
    }
    output[i] = max;
    indices[i] = (float)indx_max+1;
  }
}

__global__ void cuda_graph_pool_bprop(float* gradInput, const float *gradOutput, const float* indices, const int nClusters, const int dim) {

  extern __shared__ float shared_mem[];
  float* gradOutput_data = (float*)shared_mem;
  float* indices_data = (float*)&gradOutput_data[nClusters];

  const int tidx = threadIdx.x;
  gradInput += blockIdx.x * dim;
  gradOutput += blockIdx.x * nClusters;
  indices += blockIdx.x * nClusters;
  __syncthreads();
  gradOutput_data[tidx] = gradOutput[tidx];
  indices_data[tidx] = indices[tidx];
  __syncthreads();

  //awful i know...
  if (tidx == 1) {
    for (int i = 0; i < nClusters; ++i) {
      gradInput[(int)indices_data[i]-1] += gradOutput[i];
    }
  }
  //gradInput[(int)indices_data[tidx]-1] = gradOutput[tidx];
}

     
// input is n x d
// clusters is (d/p) x p
// output is d/p
void graph_pool_fprop_call(const float* input, const float* clusters, float* output, float* indices,
                const int nMaps,
                           const int dim, const int poolsize, const int nClusters) {

  //assert(dim % poolsize == 0);
  //const int nClusters = dim / poolsize;
  const int max_threads = 1024;
  const int threadsPerBlock = min(nClusters, max_threads);
  const int nClustersPerThread = max(DIVUP(nClusters, max_threads),1);
  dim3 threads(threadsPerBlock, 1);
  dim3 blocks(nMaps);
  int size = dim*sizeof(float) + nClusters*poolsize*sizeof(float);
  //  printf("nMaps = %d, threadsPerBlock = %d, nClustersPerThread = %d\n", nMaps, threadsPerBlock, nClustersPerThread);
  //printf("dim = %d, poolsize = %d, nClusters = %d\n", dim, poolsize, nClusters);
  cuda_graph_pool_fprop<<<blocks, threads, size>>>(input, clusters, output, indices, nMaps, dim, poolsize, nClusters, nClustersPerThread);
  CUDA_LOOK_FOR_ERROR();
}


void graph_pool_bprop_call(float* gradInput, const float* gradOutput, const float* maxIndices, 
                           const int nMaps, const int dim, const int nClusters) {
  
  const int max_threads = 1024;
  const int threadsPerBlock = min(nClusters, max_threads);
  const int nClustersPerThread = max(DIVUP(nClusters, max_threads),1);
  dim3 threads(threadsPerBlock, 1);
  dim3 blocks(nMaps);
  int size = 2*nClusters*sizeof(float);
  cuda_graph_pool_bprop<<<blocks, threads, size>>>(gradInput, gradOutput, maxIndices, nClusters, dim);
  CUDA_LOOK_FOR_ERROR();
}




static int graph_pool_fprop(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *clusterIndx = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *maxIndices = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
    
	luaL_argcheck(L, input->nDimension == output->nDimension, 2, "input and output should have same number of dimensions");
	luaL_argcheck(L, maxIndices->nDimension == output->nDimension, 2, "indices and output should have same number of dimensions");
	luaL_argcheck(L, clusterIndx->nDimension == 2, 2, "cluster_indx should be 2D tensor");

    const int nDim = input->nDimension;
    const int dim = input->size[nDim-1];
    const int nInputMaps = input->size[nDim-2];
    const int poolsize = clusterIndx->size[1];
    int nClusters = output->size[nDim-1];
    long nMaps, nSamples;
    bool resize = false;
    //fassert(nClusters == dim/poolsize);
    // haven't tested large inputs yet
    assert(dim <= 4096);

    if (nDim == 3) {
      resize = true;
      nSamples = input->size[0];
      nMaps = nSamples*nInputMaps;
      THCudaTensor_resize2d(input, nMaps, dim);
      THCudaTensor_resize2d(output, nMaps, nClusters);
      THCudaTensor_resize2d(maxIndices, nMaps, nClusters);
    }
    else {
      nMaps = nInputMaps;
    }
  
	float *input_data = (float*)THCudaTensor_data(input);
	float *output_data = (float*)THCudaTensor_data(output);
	float *clusterIndx_data = (float*)THCudaTensor_data(clusterIndx);
	float *maxIndices_data = (float*)THCudaTensor_data(maxIndices);

    graph_pool_fprop_call(input_data, clusterIndx_data, output_data, maxIndices_data, nMaps, dim, poolsize, nClusters);
    
    if (resize) {
      THCudaTensor_resize3d(input, nSamples, nInputMaps, dim);
      THCudaTensor_resize3d(output, nSamples, nInputMaps, nClusters);
      THCudaTensor_resize3d(maxIndices, nSamples, nInputMaps, nClusters);
    }
    return 0;
}


static int graph_pool_bprop(lua_State *L) {
	THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");   
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *maxIndices = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

    const int nDim = gradInput->nDimension;
    const int dim = gradInput->size[nDim-1];
    const int nClusters = gradOutput->size[nDim-1];
    const int nInputMaps = gradInput->size[nDim-2];
    long nMaps, nSamples;
    bool resize = false;
	luaL_argcheck(L, gradOutput->nDimension == nDim, 2, "gradInput and gradOutput should have same number of dimensions");
	luaL_argcheck(L, maxIndices->nDimension == nDim, 2, "indices and gradOutput should have same number of dimensions");

    if (nDim == 3) {
      resize = true;
      nSamples = gradInput->size[0];
      nMaps = nInputMaps*nSamples;
      THCudaTensor_resize2d(gradInput, nMaps, dim); 
      THCudaTensor_resize2d(gradOutput, nMaps, nClusters);
      THCudaTensor_resize2d(maxIndices, nMaps, nClusters);
    }
    else {
      nMaps = nInputMaps;
    }

	float *gradInput_data = (float*)THCudaTensor_data(gradInput);
	float *gradOutput_data = (float*)THCudaTensor_data(gradOutput);
	float *maxIndices_data = (float*)THCudaTensor_data(maxIndices);

    graph_pool_bprop_call(gradInput_data, gradOutput_data, maxIndices_data, nMaps, dim, nClusters);

    if (resize) {
      THCudaTensor_resize3d(gradInput, nSamples, nInputMaps, dim); 
      THCudaTensor_resize3d(gradOutput, nSamples, nInputMaps, nClusters);
      THCudaTensor_resize3d(maxIndices, nSamples, nInputMaps, nClusters);
    }
    return 0;
}


























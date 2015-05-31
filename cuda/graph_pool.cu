#include "cuda_common.h"
#include <cassert>

__global__ void cuda_graph_maxpool_fprop(const float* input, const float* clusters, float* output, float* indices,
                                      const int nMaps, const int dim, const int poolsize, const int nClusters, const int nClustersPerThread, const int nClustersPerBlock) {

  extern __shared__ float shared_mem[];
  float* input_data = (float*)shared_mem;
  float* cluster_indx = (float*)&input_data[dim];
  const int nThreadsPerBlock = blockDim.x;
  const int tidx = threadIdx.x;

  //  if (threadIdx.x * nClustersPerThread * poolsize >= nClusters) 
  if (threadIdx.x * nClustersPerThread >= nClusters) 
    return;

  // shift maps
  input += blockIdx.x * dim;
  output += blockIdx.x * nClusters;
  indices += blockIdx.x * nClusters;
  __syncthreads();

  output += blockIdx.y * nClustersPerBlock;
  indices += blockIdx.y * nClustersPerBlock;
  clusters += blockIdx.y * nClustersPerBlock * poolsize;
  
  __syncthreads();
  // copy input data to shared memory
  int nChunks = DIVUP(dim, nThreadsPerBlock);
  int idx;
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nThreadsPerBlock;
    if (idx < dim)
      input_data[idx] = input[idx];
  }
  __syncthreads();
  
  // copy cluster indices to shared memory 
  const int nClusterIndices = nClustersPerBlock * poolsize;
  nChunks = DIVUP(nClusterIndices,nThreadsPerBlock);
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nThreadsPerBlock;
    //    if (idx < nClusterIndices) 
    if ((idx + blockIdx.y * nClustersPerBlock * poolsize) < nClusters * poolsize)
      cluster_indx[idx] = clusters[idx]-1.0;
  }
  __syncthreads();
  

  cluster_indx += threadIdx.x * nClustersPerThread * poolsize;
  output += threadIdx.x * nClustersPerThread;
  indices += threadIdx.x * nClustersPerThread;

  __syncthreads();
  float max; 
  int indx, indx_max;
  for (int i = 0; i < nClustersPerThread; ++i) {
    if (tidx*nClustersPerThread + blockIdx.y*nClustersPerBlock < nClusters) {
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
}



__global__ void cuda_graph_avgpool_fprop(const float* input, const float* clusters, float* output, float* indices,
                                      const int nMaps, const int dim, const int poolsize, const int nClusters, const int nClustersPerThread, const int nClustersPerBlock) {

  extern __shared__ float shared_mem[];
  float* input_data = (float*)shared_mem;
  float* cluster_indx = (float*)&input_data[dim];
  const int nThreadsPerBlock = blockDim.x;
  const int tidx = threadIdx.x;

  if (threadIdx.x * nClustersPerThread >= nClusters) 
    return;

  // shift maps
  input += blockIdx.x * dim;
  output += blockIdx.x * nClusters;
  indices += blockIdx.x * nClusters;
  __syncthreads();

  output += blockIdx.y * nClustersPerBlock;
  indices += blockIdx.y * nClustersPerBlock;
  clusters += blockIdx.y * nClustersPerBlock * poolsize;
  
  __syncthreads();
  // copy input data to shared memory
  int nChunks = DIVUP(dim, nThreadsPerBlock);
  int idx;
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nThreadsPerBlock;
    if (idx < dim)
      input_data[idx] = input[idx];
  }
  __syncthreads();
  
  // copy cluster indices to shared memory 
  const int nClusterIndices = nClustersPerBlock * poolsize;
  nChunks = DIVUP(nClusterIndices,nThreadsPerBlock);
  for (int i = 0; i < nChunks; ++i) {
    idx = tidx + i*nThreadsPerBlock;
    if ((idx + blockIdx.y * nClustersPerBlock * poolsize) < nClusters * poolsize)
      cluster_indx[idx] = clusters[idx]-1.0;
  }
  __syncthreads();
  

  cluster_indx += threadIdx.x * nClustersPerThread * poolsize;
  output += threadIdx.x * nClustersPerThread;
  indices += threadIdx.x * nClustersPerThread;

  __syncthreads();
  float sum = 0; 
  int indx;
  for (int i = 0; i < nClustersPerThread; ++i) {
    if (tidx*nClustersPerThread + blockIdx.y*nClustersPerBlock < nClusters) {
      for (int j = 0; j < poolsize; ++j) {
        indx = (int)cluster_indx[i*poolsize + j];
        sum += input_data[indx];
      }
      output[i] = sum/poolsize;
    }
  }
}

__global__ void cuda_graph_avgpool_bprop(float* gradInput, const float *gradOutput, const float* clusters, const int nClusters, const int poolsize, const int dim, const int nClustersPerThread) {

  extern __shared__ float shared_mem[];
  float* gradOutput_data = (float*)shared_mem;

  const int tidx = threadIdx.x;
  gradInput += blockIdx.x * dim;
  gradOutput += blockIdx.x * nClusters;
  __syncthreads();
  for (int i = 0; i < nClustersPerThread; ++i) {
    int idx = tidx + i*blockDim.x;
      if (idx < nClusters) {
        gradOutput_data[idx] = gradOutput[idx];
      }
  }
  __syncthreads();


  if (tidx < poolsize) {
    for (int i = 0; i < nClusters; ++i) {
        gradInput[(int)(clusters[i*poolsize+tidx]-1)] += gradOutput[i]/poolsize;
    }
  }

  /*
  for (int j = 0; j < poolsize; ++j) {
    gradInput[(int)(clusters[tidx*poolsize+j]-1)] += gradOutput[tidx]/poolsize;
    __syncthreads();
  }
  */
  __syncthreads();

  /*
  //ouch...
  if (tidx == 1) {
    for (int i = 0; i < nClusters; ++i) {
      //    int idx = tidx + i*blockDim.x;
      for (int j = 0; j < poolsize; ++j) {
        gradInput[(int)(clusters[i*poolsize+j]-1)] += gradOutput[i]/poolsize;
      }
    }
  }
  */




  /*
  for (int i = 0; i < nClustersPerThread; ++i) {
    int idx = tidx + i*blockDim.x;
      if (idx < nClusters) {
        for (int j = 0; j < poolsize; ++j) {
          gradInput[(int)clusters[idx*poolsize+j]] += gradOutput_data[idx]/poolsize;
      }
  }
  }
  */
}

__global__ void cuda_graph_maxpool_bprop(float* gradInput, const float *gradOutput, const float* indices, const int nClusters, const int dim, const int nClustersPerThread) {

  extern __shared__ float shared_mem[];
  float* gradOutput_data = (float*)shared_mem;
  float* indices_data = (float*)&gradOutput_data[nClusters];

  const int tidx = threadIdx.x;
  gradInput += blockIdx.x * dim;
  gradOutput += blockIdx.x * nClusters;
  indices += blockIdx.x * nClusters;
  __syncthreads();
  for (int i = 0; i < nClustersPerThread; ++i) {
    int idx = tidx + i*blockDim.x;
      if (idx < nClusters) {
        gradOutput_data[idx] = gradOutput[idx];
        indices_data[idx] = indices[idx];
      }
  }
  __syncthreads();

  //ouch...
  if (tidx == 1) {
    for (int i = 0; i < nClusters; ++i) {
      gradInput[(int)indices_data[i]-1] += gradOutput[i];
    }
  }
  //gradInput[(int)indices_data[tidx]-1] = gradOutput[tidx];
}

     
// input is nMaps x dim
// clusters is poolsize x nClusters
// output is nMaps x nClusters 
// indices is nMaps x nClusters
void graph_pool_fprop_call(const float* input, const float* clusters, float* output, float* indices,
                const int nMaps,
                           const int dim, const int poolsize, const int nClusters, const int type) {

  const int max_threads = 1024;
  const int max_clusters_per_block = 200; //floor((49000/sizeof(float) - dim)/poolsize);
  const int nBlocksPerMap = max(DIVUP(nClusters,max_clusters_per_block),1); 
  const int nClustersPerBlock = max(DIVUP(nClusters, nBlocksPerMap),1);
  const int nClustersPerThread = max(DIVUP(nClustersPerBlock, max_threads),1);
  const int threadsPerBlock = min(nClustersPerBlock, max_threads);
  dim3 threads(threadsPerBlock, 1);
  dim3 blocks(nMaps, nBlocksPerMap);

  int size = dim*sizeof(float) + nClustersPerBlock*poolsize*sizeof(float);
    //    printf("nMaps = %d, nClusters = %d, threadsPerBlock = %d, nClustersPerThread = %d\n", nMaps, nClusters, threadsPerBlock, nClustersPerThread);
    //printf("dim = %d, poolsize = %d, nClusters = %d, size = %d\n", dim, poolsize, nClusters, size);
    //printf("nClustersPerBlock = %d, nBlocksPerMap = %d\n",nClustersPerBlock, nBlocksPerMap);
  if (type == 1)
    cuda_graph_maxpool_fprop<<<blocks, threads, size>>>(input, clusters, output, indices, nMaps, dim, poolsize, nClusters, nClustersPerThread, nClustersPerBlock);
  else
    cuda_graph_avgpool_fprop<<<blocks, threads, size>>>(input, clusters, output, indices, nMaps, dim, poolsize, nClusters, nClustersPerThread, nClustersPerBlock);

  CUDA_LOOK_FOR_ERROR();
  // printf("done\n");
}


 void graph_pool_bprop_call(float* gradInput, const float* gradOutput, const float* maxIndices, const float* clusters, 
                           const int nMaps, const int dim, const int nClusters, const int poolsize, const int type) {
  
  const int max_threads = 1024;
  assert(nClusters <= max_threads);
  const int nThreadsPerMap = max(DIVUP(nClusters, max_threads),1);
  const int threadsPerBlock = min(nClusters, max_threads);
  dim3 threads(threadsPerBlock, 1);
  dim3 blocks(nMaps);
  int size = 2*nClusters*sizeof(float);
  //  printf("nThreadsPerMap = %d, threadsPerBlock = %d\n",nThreadsPerMap, threadsPerBlock);
  if (type == 1)
    cuda_graph_maxpool_bprop<<<blocks, threads, size>>>(gradInput, gradOutput, maxIndices, nClusters, dim, nThreadsPerMap);
  else
    cuda_graph_avgpool_bprop<<<blocks, threads, size>>>(gradInput, gradOutput, clusters, nClusters, poolsize, dim, nThreadsPerMap);

  CUDA_LOOK_FOR_ERROR();
}




static int graph_pool_fprop(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *clusterIndx = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *maxIndices = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
    const int type = luaL_checkint(L, 5);
    
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
    //    printf("dim = %d\n",dim);
    assert(dim <= 4096);

    if (nDim == 3) {
      resize = true;
      nSamples = input->size[0];
      nMaps = nSamples*nInputMaps;
      THCudaTensor_resize2d(NULL,input, nMaps, dim);
      THCudaTensor_resize2d(NULL,output, nMaps, nClusters);
      THCudaTensor_resize2d(NULL,maxIndices, nMaps, nClusters);
    }
    else {
      nMaps = nInputMaps;
    }
  
	float *input_data = (float*)THCudaTensor_data(NULL,input);
	float *output_data = (float*)THCudaTensor_data(NULL,output);
	float *clusterIndx_data = (float*)THCudaTensor_data(NULL,clusterIndx);
	float *maxIndices_data = (float*)THCudaTensor_data(NULL,maxIndices);

    graph_pool_fprop_call(input_data, clusterIndx_data, output_data, maxIndices_data, nMaps, dim, poolsize, nClusters, type);
    //printf("maincoons\n");
    
    if (resize) {
      THCudaTensor_resize3d(NULL,input, nSamples, nInputMaps, dim);
      THCudaTensor_resize3d(NULL,output, nSamples, nInputMaps, nClusters);
      THCudaTensor_resize3d(NULL,maxIndices, nSamples, nInputMaps, nClusters);
    }
    //printf("maincats\n");
    return 0;
}


static int graph_pool_bprop(lua_State *L) {
	THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");   
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *maxIndices = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *clusters = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
    const int type = luaL_checkint(L, 5);

    const int nDim = gradInput->nDimension;
    const int dim = gradInput->size[nDim-1];
    const int nClusters = gradOutput->size[nDim-1];
    const int nInputMaps = gradInput->size[nDim-2];
    const int poolsize = clusters->size[1];
    long nMaps, nSamples;
    bool resize = false;
	luaL_argcheck(L, gradOutput->nDimension == nDim, 2, "gradInput and gradOutput should have same number of dimensions");
	luaL_argcheck(L, maxIndices->nDimension == nDim, 2, "indices and gradOutput should have same number of dimensions");

    if (nDim == 3) {
      resize = true;
      nSamples = gradInput->size[0];
      nMaps = nInputMaps*nSamples;
      THCudaTensor_resize2d(NULL,gradInput, nMaps, dim); 
      THCudaTensor_resize2d(NULL,gradOutput, nMaps, nClusters);
      THCudaTensor_resize2d(NULL,maxIndices, nMaps, nClusters);
    }
    else {
      nMaps = nInputMaps;
    }

	float *gradInput_data = (float*)THCudaTensor_data(NULL,gradInput);
	float *gradOutput_data = (float*)THCudaTensor_data(NULL,gradOutput);
	float *maxIndices_data = (float*)THCudaTensor_data(NULL,maxIndices);
	float *clusters_data = (float*)THCudaTensor_data(NULL,clusters);

    graph_pool_bprop_call(gradInput_data, gradOutput_data, maxIndices_data, clusters_data, nMaps, dim, nClusters, poolsize, type);

    if (resize) {
      THCudaTensor_resize3d(NULL,gradInput, nSamples, nInputMaps, dim); 
      THCudaTensor_resize3d(NULL,gradOutput, nSamples, nInputMaps, nClusters);
      THCudaTensor_resize3d(NULL,maxIndices, nSamples, nInputMaps, nClusters);
    }
    return 0;
}

#include "luaT.h"
#include "THC/THC.h"
#include "arithmetic.h"

__global__ void modulus_updateGradInput_kernel(float* input, float* output, float* gradInput, float* gradOutput, int n) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n)
    return;
  const float eps = 0.0001;
  const float c = gradOutput[i]/max(output[i],eps);
  gradInput[2*i] = input[2*i]*c;
  gradInput[2*i+1] = input[2*i+1]*c;
}

int modulus_updateGradInput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradInput =(THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradOutput =(THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  
  const long nSamples = input->size[0];
  const long nInputPlanes = input ->size[1];
  const long iH = input->size[2];
  const long iW = input->size[3];
  const long nOutputPlanes = gradOutput->size[1];
  const long n = nSamples*nInputPlanes*iH*iW;
  
  THCudaTensor_resize2d(input,n,2);
  THCudaTensor_resize2d(gradInput,n,2);
  THCudaTensor_resize1d(output,n);
  THCudaTensor_resize1d(gradOutput,n);

  float* input_data = THCudaTensor_data(input);
  float* output_data = THCudaTensor_data(output);
  float* gradInput_data = THCudaTensor_data(gradInput);
  float* gradOutput_data = THCudaTensor_data(gradOutput);

  const int numsPerBlock = 64;
  dim3 threads(numsPerBlock,2);
  dim3 blocks(DIVUP(n,numsPerBlock),1,1);
  modulus_updateGradInput_kernel<<<blocks,threads>>>(input_data, output_data, gradInput_data, gradOutput_data, n);
  
  THCudaTensor_resize5d(input,nSamples,nInputPlanes,iH,iW,2);
  THCudaTensor_resize5d(gradInput,nSamples,nInputPlanes,iH,iW,2);
  THCudaTensor_resize4d(output,nSamples,nOutputPlanes,iH,iW);
  THCudaTensor_resize4d(gradOutput,nSamples,nOutputPlanes,iH,iW);

  return 0;
}


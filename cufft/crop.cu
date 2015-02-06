#include "luaT.h"
#include "THC/THC.h"
#include "arithmetic.h"

__global__ void batch_crop_kernel(float* input,
                           const int nCropRows, const int nCropCols, 
                                         const int iH, const int iW,
                                         const int nPlanes){
  const int plane = blockIdx.x;
  if (plane >= nPlanes)
    return;

  input += plane * iH * iW;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  if (ty < iH && (ty > iH-nCropRows-1 || ty < nCropRows)) {
    input[ty*iW + tx] = 0;
  }
  if (tx < iW && (tx > iW-nCropCols-1 || tx < nCropCols)) {
    input[ty*iW + tx] = 0;
  }
}


// we are assuming the input is real, not complex
static int crop_zeroborders(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  const int nCropRows = luaL_checknumber(L,2);
  const int nCropCols = luaL_checknumber(L,3);

  const int dim = input->nDimension;
  const int iH = input->size[dim-2];
  const int iW = input->size[dim-1];

  long nPlanes, nInputPlanes, nOutputPlanes;
  bool resize = false;

  if (dim == 4) {
    resize = true;
    nOutputPlanes = input->size[0];
    nInputPlanes = input->size[1];
    nPlanes = nInputPlanes*nOutputPlanes;
    THCudaTensor_resize3d(input, nPlanes, iH, iW);
  }
  else {
    nPlanes = input->size[0];
  }

  float* input_data = (float*)THCudaTensor_data(input);
  assert(iH == iW);

  dim3 threads(iH,iW);
  dim3 blocks(nPlanes);
  batch_crop_kernel<<<blocks, threads>>>(input_data,
                                         nCropRows, nCropCols, 
                                         iH, iW, nPlanes);

  if (resize) {
    THCudaTensor_resize4d(input, nOutputPlanes, nInputPlanes, iH, iW);
  }

  CUDA_LOOK_FOR_ERROR();
  return 0;
}




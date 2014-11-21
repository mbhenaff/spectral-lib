#include "luaT.h"
#include "_bias.cu"
#include "THC.h"


int bias_updateOutput(lua_State *L) {
  THCudaTensor *bias =
    (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  luaL_argcheck(L, (input->nDimension == 3) || (input->nDimension == 4), 2,
		"3D or 4D (batch mode) tensor is expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(input), 2,
		"input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(bias), 1,
		"bias must be contiguous");

  const int ndim = input->nDimension;
  int dimp = 0, dimh = 1, dimw = 2;
  if (ndim == 4) {
    ++dimp; ++dimh; ++dimw;
  }


  long batchSize = (ndim == 4) ? input->size[0] : 1;
  const long nPlanes = input->size[dimp];
  const long iH = input->size[dimh];
  const long iW = input->size[dimw];

  float* input_p  = THCudaTensor_data(input);
  const float* bias_p   = THCudaTensor_data(bias);
  
  _add_bias<<<dim3(nPlanes), dim3(32, 4)>>>(bias_p, input_p,
						 batchSize, nPlanes,
                         iW, iH);
  return 0;
}



int bias_accGradParameters(lua_State *L) {
  THCudaTensor *gradBias =
    (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float scale = luaL_optnumber(L, 3, 1.f);

  luaL_argcheck(L, (gradOutput->nDimension == 3) || (gradOutput->nDimension == 4), 2,
		"3D or 4D (batch mode) tensor is expected");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradOutput), 3,
		"gradOutput must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(gradBias), 1,
		"gradBias must be contiguous");

  const int ndim = gradOutput->nDimension;
  int dimp = 0, dimh = 1, dimw = 2;
  if (ndim == 4) {
    ++dimp; ++dimh; ++dimw;
  }

  long batchSize = (ndim == 4) ? gradOutput->size[0] : 1;
  const long nPlanes = gradOutput->size[dimp];
  const long iH = gradOutput->size[dimh];
  const long iW = gradOutput->size[dimw];
  const float* gradOutput_p = THCudaTensor_data(gradOutput);
  float*       gradBias_p   = THCudaTensor_data(gradBias);
  
  _fill_gradBias<<<dim3(nPlanes), dim3(32, 4)>>>(gradBias_p, gradOutput_p,
						       scale, batchSize,
						       nPlanes, iW, iH);
  return 0;
}

#include "luaT.h"
#include "THC/THC.h"
#include <cufft.h>
#include "fft_prod2.cu"
#include "fill_hermitian.cu"
#include "modulus.cu"
#include "complexInterp.cu"
#include "bias.cu"
#include "crop.cu"
#include "prod.cu"
#include "graph_pool.cu"

#include "fft_prod_module.cu"
#include "cufft.cpp"

static int prod_fprop_real(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *weight = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 3D tensor");
	luaL_argcheck(L, output->nDimension == 3, 2, "output should be 3D tensor");
	luaL_argcheck(L, weight->nDimension == 3, 2, "kernel should be 3D tensor");

	long nMinibatch = input->size[0];
	long nOutputMaps = weight->size[0];
	long nInputMaps = weight->size[1];
	long dim = input->size[2];

	// raw pointers
	float *input_data = (float*)THCudaTensor_data(NULL,input);
	float *weight_data = (float*)THCudaTensor_data(NULL,weight);
	float *output_data = (float*)THCudaTensor_data(NULL,output);
	
	spectral_prod(input_data, weight_data, output_data, dim,
				nMinibatch, nInputMaps*dim, nOutputMaps*dim,
				nInputMaps, dim, dim, 
                nOutputMaps, nInputMaps*dim, dim);

	return 0;
}


static int prod_bprop_real(lua_State *L) {
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *weight = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

	luaL_argcheck(L, gradInput->nDimension == 3, 2, "gradInput should be 3D tensor");
	luaL_argcheck(L, weight->nDimension == 3, 2, "weight should be 3D tensor");
	luaL_argcheck(L, gradOutput->nDimension == 3, 2, "gradOutput should be 3D tensor");

	long nMinibatch = gradInput->size[0];
	long nOutputMaps = weight->size[0];
	long nInputMaps = weight->size[1];
	long dim = gradInput->size[2];

	// raw pointers
	float *gradOutput_data = (float*)THCudaTensor_data(NULL,gradOutput);
	float *weight_data = (float*)THCudaTensor_data(NULL,weight);
	float *gradInput_data = (float*)THCudaTensor_data(NULL,gradInput);
	
	spectral_prod(gradOutput_data, weight_data, gradInput_data, dim,
				nMinibatch, nOutputMaps*dim, nInputMaps*dim,
				nOutputMaps, dim, dim*nInputMaps, 
                nInputMaps, dim, dim);

	return 0;
}



static int prod_accgrad_real(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradWeight = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 3D tensor");
	luaL_argcheck(L, gradOutput->nDimension == 3, 2, "gradOutput should be 3D tensor");
	luaL_argcheck(L, gradWeight->nDimension == 3, 2, "gradWeight should be 3D tensor");

	long nMinibatch = input->size[0];
	long nOutputMaps = gradWeight->size[0];
	long nInputMaps = gradWeight->size[1];
	long dim = input->size[2];

	// raw pointers
	float *input_data = (float*)THCudaTensor_data(NULL,input);
	float *gradOutput_data = (float*)THCudaTensor_data(NULL,gradOutput);
	float *gradWeight_data = (float*)THCudaTensor_data(NULL,gradWeight);
	
	spectral_prod(input_data, gradOutput_data, gradWeight_data, dim,
				nInputMaps, dim, dim,
				nMinibatch, nInputMaps*dim, nOutputMaps*dim, 
                nOutputMaps, dim, nInputMaps*dim);
	return 0;
}


static int fill_hermitian(lua_State *L) {
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

  luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 1, "input must be contiguous");
  luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output),2, "output must be contiguous");
  luaL_argcheck(L, input->nDimension == 5, 2, "input should be 4D complex tensor");
  luaL_argcheck(L, output->nDimension == 5, 2, "output should be 4D complex tensor");
  luaL_argcheck(L, input->size[3] == output->size[3]/2+1, 2, "input must have N/2+1 columns");

  long nMinibatch = input->size[0];
  long nInputPlanes = input->size[1];
  long nRows = output->size[2];
  long nCols = output->size[3];
  cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
  cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
   
  fill_hermitian_call(input_data, output_data, nMinibatch*nInputPlanes,nRows,nCols);

  return 0;
}

static const struct luaL_reg libspectralnet_init [] = {
  {"fft1d_r2c", fft1d_r2c},
  {"fft1d_c2r", fft1d_c2r},
  {"fft1d_c2c", fft1d_c2c},
  {"fft2d_r2c", fft2d_r2c},
  {"fft2d_c2r", fft2d_c2r},
  {"fft2d_c2c", fft2d_c2c},
  {"prod_fprop_real", prod_fprop_real},
  {"prod_bprop_real", prod_bprop_real},
  {"prod_accgrad_real", prod_accgrad_real},
  {"prod_fprop_complex", prod_fprop_complex},
  {"prod_bprop_complex", prod_bprop_complex},
  {"prod_accgrad_complex",prod_accgrad_complex},
  {"fill_hermitian",fill_hermitian},
  {"modulus_updateGradInput",modulus_updateGradInput},
  {"complexInterp_interpolate",complexInterp_interpolate},
  {"bias_updateOutput", bias_updateOutput},
  {"bias_accGradParameters", bias_accGradParameters},
  {"crop_zeroborders",crop_zeroborders},
  {"graph_pool_fprop", graph_pool_fprop},
  {"graph_pool_bprop", graph_pool_bprop},
  {NULL, NULL}
};

LUA_EXTERNC int luaopen_libspectralnet(lua_State *L) {
	luaL_openlib(L,"libspectralnet",libspectralnet_init,0);
	return 1;
}





	

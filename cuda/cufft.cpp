#include "luaT.h"
#include "THC/THC.h"
#include <cufft.h>


static int fft1d_r2c(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

	luaL_argcheck(L, input->nDimension == 2, 2, "input should be 2D real tensor [nLines x N]");
	luaL_argcheck(L, output->nDimension == 3, 2, "output should be 2D complex tensor [nLines x (N/2+1) x 2]");
	
	long nInputLines = input->size[0];
	long N = input->size[1];

	// argument check
	luaL_argcheck(L, output->size[0] == nInputLines, 0, "input and output should have the same number of lines");
	luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
	luaL_argcheck(L, output->size[1] == N/2+1, 0, "output should be N/2+1");
	luaL_argcheck(L, output->size[2] == 2, 0, "output should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	

	// raw pointers 
	float *input_data = THCudaTensor_data(NULL,input);
	cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
	
	// execute FFT
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_R2C, nInputLines);
	cufftExecR2C(plan, (cufftReal*)input_data, (cufftComplex*)output_data);

	// clean up
	cufftDestroy(plan);

	return 0;	
}


static int fft1d_c2r(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

	luaL_argcheck(L, output->nDimension == 2, 2, "output should be 2D real tensor [nLines x N]");
   	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 2D complex tensor [nLines x (N/2+1) x 2]");
	
	long nInputLines = input->size[0];
	long N = output->size[1];

	// argument check
	luaL_argcheck(L, output->size[0] == nInputLines, 0, "input and output should have the same number of lines");
	luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
	luaL_argcheck(L, input->size[1] == N/2+1, 0, "input should be N/2+1");
	luaL_argcheck(L, input->size[2] == 2, 0, "input should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	

	// raw pointers 
	float *output_data = THCudaTensor_data(NULL,output);
	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	
	// execute FFT
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_C2R, nInputLines);
	cufftExecC2R(plan, (cufftComplex*)input_data, (cufftReal*)output_data);

	// clean up
	cufftDestroy(plan);
	return 0;	
}


static int fft1d_c2c(lua_State *L) {	
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	int dir = luaL_checkint(L,3);

	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 2D (batch) complex tensor");
	luaL_argcheck(L, output->nDimension == 3, 2, "output should be 2D (batch) complex tensor");
	luaL_argcheck(L, dir == 1 || dir == -1, 2, "direction should be 1 or -1");
	
	long nInputLines = input->size[0];
	long N = input->size[1];

	// argument check
	luaL_argcheck(L, output->size[0] == nInputLines, 0, "input and output should have the same number of lines");
	luaL_argcheck(L, output->size[1] == N, 0, "input and output should have the same dimension");
	luaL_argcheck(L, (output->size[2] == 2) && (input->size[2] == 2), 0, "input and output should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	
	// raw pointers 
	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
	
	// execute FFT
	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_C2C, nInputLines);
	cufftExecC2C(plan, (cufftComplex*)input_data, (cufftComplex*)output_data,-dir);

	// clean up
	cufftDestroy(plan);
	return 0;	
}




static int fft2d_r2c(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

	luaL_argcheck(L, input->nDimension == 3, 2, "input should be 3D real tensor [nPlanes x N x M]");
	luaL_argcheck(L, output->nDimension == 4, 2, "output should be 3D complex tensor [nPlanes x (N/2+1) x (M/2+1) x 2]");
	
	long nInputPlanes = input->size[0];
	long N = input->size[1];
	long M = input-> size[2];
    int size[2] = {N,M};

	// argument check
	luaL_argcheck(L, output->size[0] == nInputPlanes, 0, "input and output should have the same number of planes");
	luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
	luaL_argcheck(L, (M % 2) == 0, 0, "M should be multiple of 2");
	//luaL_argcheck(L, output->size[1] == N/2+1, 0, "output should be N/2+1");
	luaL_argcheck(L, output->size[2] == M/2+1, 0, "output should be M/2+1");
	luaL_argcheck(L, output->size[3] == 2, 0, "output should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	

	// raw pointers 
	float *input_data = THCudaTensor_data(NULL,input);
	cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
	
	// execute FFT
	cufftHandle plan;
	cufftPlanMany(&plan, 2, size, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, nInputPlanes);
    cufftExecR2C(plan, (cufftReal*)input_data, (cufftComplex*)output_data);

	// clean up
	cufftDestroy(plan);

	return 0;	
}



static int fft2d_c2r(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

	luaL_argcheck(L, output->nDimension == 3, 2, "input should be 3D real tensor [nPlanes x N x M]");
	luaL_argcheck(L, input->nDimension == 4, 2, "output should be 3D complex tensor [nPlanes x (N/2+1) x (M/2+1) x 2]");
	
	long nInputPlanes = input->size[0];
	long N = output->size[1];
	long M = output-> size[2];
    int size[2] = {N,M};

	// argument check
	luaL_argcheck(L, output->size[0] == nInputPlanes, 0, "input and output should have the same number of planes");
	luaL_argcheck(L, (N % 2) == 0, 0, "N should be multiple of 2");
	luaL_argcheck(L, (M % 2) == 0, 0, "M should be multiple of 2");
	//luaL_argcheck(L, input->size[1] == N/2+1, 0, "output should be N/2+1");
	luaL_argcheck(L, input->size[2] == M/2+1, 0, "output should be M/2+1");
	luaL_argcheck(L, input->size[3] == 2, 0, "output should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 2, "output must be contiguous");
	
	// raw pointers 
	float *output_data = THCudaTensor_data(NULL,output);
	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	
	// execute FFT
	cufftHandle plan;
	cufftPlanMany(&plan, 2, size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, nInputPlanes);
	cufftExecC2R(plan, (cufftComplex*)input_data, (cufftReal*)output_data);

	// clean up
	cufftDestroy(plan);

	return 0;	
}


static int fft2d_c2c(lua_State *L) {	
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	int dir = luaL_checkint(L,3);

	luaL_argcheck(L, input->nDimension == 4, 2, "input should be 3D (batch) complex tensor");
	luaL_argcheck(L, output->nDimension == 4, 2, "output should be 3D (batch) complex tensor");
	luaL_argcheck(L, dir == 1 || dir == -1, 2, "direction should be 1 or -1");
	
	long nInputPlanes = input->size[0];
	long N = input->size[1];
	long M = input->size[2];
    int size[2] = {N,M};

	// argument check
	luaL_argcheck(L, output->size[0] == nInputPlanes, 0, "input and output should have the same number of planes");
	luaL_argcheck(L, output->size[1] == N, 0, "input and output should have the same dimension");
	luaL_argcheck(L, output->size[2] == M, 0, "input and output should have the same dimension");
	luaL_argcheck(L, (output->size[3] == 2) && (input->size[3] == 2), 0, "input and output should be complex");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,input), 2, "input must be contiguous");
	luaL_argcheck(L, THCudaTensor_isContiguous(NULL,output), 1, "output must be contiguous");
	// raw pointers 
	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
    
	// execute FFT
	cufftHandle plan;
	cufftPlanMany(&plan, 2, size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, nInputPlanes);
	cufftExecC2C(plan, (cufftComplex*)input_data, (cufftComplex*)output_data,-dir);

	// clean up
	cufftDestroy(plan);
	return 0;	
}



// table which contains names of functions in Lua and C
static const struct luaL_reg libcufft [] = {
  {"fft1d_r2c", fft1d_r2c},
  {"fft1d_c2r", fft1d_c2r},
  {"fft1d_c2c", fft1d_c2c},
  {"fft2d_r2c", fft2d_r2c},
  {"fft2d_c2r", fft2d_c2r},
  {"fft2d_c2c", fft2d_c2c},
  {NULL, NULL}
};


LUA_EXTERNC int luaopen_libcufft(lua_State *L) {
	luaL_openlib(L, "libcufft", libcufft, 0);
	return 1;
}



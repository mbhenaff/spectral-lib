#include "fft_product2.cu"


static int prod_fprop_complex(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *weight = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    bool conjWeight = lua_toboolean(L,4);
    bool accumulate = false;

	luaL_argcheck(L, input->nDimension == 5, 2, "input should be 4D complex tensor");
	luaL_argcheck(L, output->nDimension == 5, 2, "output should be 4D complex tensor");
	luaL_argcheck(L, weight->nDimension == 5, 2, "kernel should be 4D complex tensor");

	long nMinibatch = input->size[0];
	long nOutputPlanes = weight->size[0];
	long nInputPlanes = weight->size[1];
	long nRows = input->size[2];
	long nCols = input->size[3];
    long planeSize = nRows*nCols;

	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	cuComplex *weight_data = (cuComplex*)THCudaTensor_data(NULL,weight);
	cuComplex *output_data = (cuComplex*)THCudaTensor_data(NULL,output);
	
	fft_product_call(input_data, weight_data, output_data, nRows, nCols,
				nMinibatch, nInputPlanes*planeSize, nOutputPlanes*planeSize,
				nInputPlanes, planeSize, planeSize, 
                nOutputPlanes, nInputPlanes*planeSize, planeSize, 
                accumulate, conjWeight);

	return 0;
}

static int prod_bprop_complex(lua_State *L) {
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *weight = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    bool conjWeight = lua_toboolean(L, 4);
    bool accumulate = false;

	luaL_argcheck(L, gradInput->nDimension == 5, 2, "gradInput should be 4D complex tensor");
	luaL_argcheck(L, weight->nDimension == 5, 2, "weight should be 4D complex tensor");
	luaL_argcheck(L, gradOutput->nDimension == 5, 2, "gradOutput should be 4D complex tensor");

	long nMinibatch = gradInput->size[0];
	long nOutputPlanes = weight->size[0];
	long nInputPlanes = weight->size[1];
	long nRows = gradInput->size[2];
	long nCols = gradInput->size[3];

	cuComplex *gradOutput_data = (cuComplex*)THCudaTensor_data(NULL,gradOutput);
	cuComplex *weight_data = (cuComplex*)THCudaTensor_data(NULL,weight);
	cuComplex *gradInput_data = (cuComplex*)THCudaTensor_data(NULL,gradInput);
	
	fft_product_call(gradOutput_data, weight_data, gradInput_data, nRows, nCols,
				nMinibatch, nOutputPlanes*nRows*nCols, nInputPlanes*nRows*nCols,
				nOutputPlanes, nRows*nCols, nRows*nCols*nInputPlanes, 
                nInputPlanes, nRows*nCols, nRows*nCols, 
                accumulate, conjWeight);

	return 0;
}

static int prod_accgrad_complex(lua_State *L) {
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");	
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradWeight = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
    bool conjGradOutput = lua_toboolean(L, 4);
    bool accumulate = false;

	luaL_argcheck(L, input->nDimension == 5, 2, "input should be 4D complex tensor");
	luaL_argcheck(L, gradOutput->nDimension == 5, 2, "gradOutput should be 4D complex tensor");
	luaL_argcheck(L, gradWeight->nDimension == 5, 2, "gradWeight should be 4D complex tensor");

	long nMinibatch = input->size[0];
	long nOutputPlanes = gradWeight->size[0];
	long nInputPlanes = gradWeight->size[1];
	long nRows = input->size[2];
	long nCols = input->size[3];

	cuComplex *input_data = (cuComplex*)THCudaTensor_data(NULL,input);
	cuComplex *gradOutput_data = (cuComplex*)THCudaTensor_data(NULL,gradOutput);
	cuComplex *gradWeight_data = (cuComplex*)THCudaTensor_data(NULL,gradWeight);
	
	fft_product_call(input_data, gradOutput_data, gradWeight_data, nRows, nCols,
				nInputPlanes, nRows*nCols, nRows*nCols,
				nMinibatch, nInputPlanes*nRows*nCols, nOutputPlanes*nRows*nCols, 
                nOutputPlanes, nRows*nCols, nInputPlanes*nRows*nCols,
                accumulate, conjGradOutput);

	return 0;
}




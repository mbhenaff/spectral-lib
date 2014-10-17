--module to perform complex interpolation on inputs. 
--real and imaginary parts are interpolated separately
--optionally we can choose the outputs to be hermitian, i.e. elements 1 and N/2+1 are real. 
--this assumes the input already represents only half of the elements (the others are assumed to be their complex conjugates)

require 'nn'
require 'interp'

local ComplexInterp, parent = torch.class('nn.ComplexInterp','nn.Module')

-- inputSize should be a tensor or table {nRows,nCols}, same for outputSize 
-- hermitianRows/hermitianCols should be true/false indicating if we assume the inputs are hermitian

function ComplexInterp:__init(inputSize,outputSize,hermitianRows,hermitianCols)
	parent.__init(self)

	self.hermitianRows = hermitianRows or false
	self.hermitianCols = hermitianCols or false
	self.inputSize = inputSize
	self.outputSize = outputSize
	self.kernelRowsReal = interpKernel(inputSize[1],outputSize[1])
	self.kernelColsReal = interpKernel(inputSize[2],outputSize[2])
	if self.hermitianRows then
		self.kernelRowsImag = self.kernelRowsReal:clone()
		self.kernelRowsImag[1]:zero()
		self.kernelRowsImag[inputSize[1]]:zero()
	else
		self.kernelRowsImag = self.kernelRowsReal
	end
	if self.hermitianCols then	
		self.kernelColsImag = self.kernelColsReal:clone()
		self.kernelColsImag[1]:zero()
		self.kernelColsImag[inputSize[2]]:zero()
	else
		self.kernelColsImag = self.kernelColsReal
	end
	self.buffer = torch.Tensor(outputSize[1],inputSize[2])
	self.buffer2 = torch.Tensor(inputSize[1],outputSize[2])
end
	

-- assuming inputs are size [dim1 x dim2 x nRows x nCols x 2]
-- or [dim1 x nRows x nCols x 2]
function ComplexInterp:updateOutput(input)
	--print('complex interp updating output')
	local ndim = input:nDimension()
	local size = input:size()
	if ndim == 5 then
		-- need this hack since reshape module only supports 4 dimensions
		input:resize(size[1]*size[2],size[3],size[4],2)
	end
	local nFrames = input:size(1)
	self.output:resize(nFrames, self.outputSize[1], self.outputSize[2], 2)	
	self.output:zero()
	for i = 1,nFrames do
		-- interpolate real part
		self.buffer:zero()
		self.buffer:addmm(self.kernelRowsReal:t(),input[i][{{},{},1}])
		self.output[i][{{},{},1}]:addmm(self.buffer,self.kernelColsReal)
		-- interpolate imaginary part
		self.buffer:zero()
		self.buffer:addmm(self.kernelRowsImag:t(),input[i][{{},{},2}])
		self.output[i][{{},{},2}]:addmm(self.buffer,self.kernelColsImag)
	end	
	if ndim == 5 then
		input:resize(size)
		self.output:resize(size[1],size[2],self.outputSize[1],self.outputSize[2],2)
	end
	return self.output
end



function ComplexInterp:updateGradInput(input, gradOutput)
	local ndim = gradOutput:nDimension()
	local size = gradOutput:size()
	if ndim == 5 then
		gradOutput:resize(size[1]*size[2],size[3],size[4],2)
	end
	local nFrames = gradOutput:size(1)
	self.gradInput:resize(nFrames, self.inputSize[1], self.inputSize[2], 2)
	self.gradInput:zero()
	for i = 1,nFrames do
		self.buffer2:zero()
		-- downsample real part
		self.buffer2:addmm(self.kernelRowsReal,gradOutput[i][{{},{},1}])
		self.gradInput[i][{{},{},1}]:addmm(self.buffer2,self.kernelColsReal:t())
		-- downsample imaginary part
		self.buffer2:zero()
		self.buffer2:addmm(self.kernelRowsImag,gradOutput[i][{{},{},2}])
		self.gradInput[i][{{},{},2}]:addmm(self.buffer2,self.kernelColsImag:t())
	end
	if ndim == 5 then
		gradOutput:resize(size)
		self.gradInput:resize(size[1],size[2],self.inputSize[1],self.inputSize[2],2)
	end
	--print('complex interp gradInput norm = ',self.gradInput:norm())
	return self.gradInput
end



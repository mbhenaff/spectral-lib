--module to perform complex interpolation on inputs. 
--real and imaginary parts are interpolated separately
--optionally we can choose the outputs to be hermitian, i.e. elements 1 and N/2+1 are real. 
--this assumes the input already represents only half of the elements (the others are assumed to be their complex conjugates)

require 'nn'
require 'interp'

local ComplexInterp, parent = torch.class('nn.ComplexInterp','nn.Module')

-- inputSize should be a tensor or table {nRows,nCols}, same for outputSize 
-- hermitianRows/hermitianCols should be true/false indicating if we assume the inputs are hermitian

function ComplexInterp:__init(iH, iW, oH, oW)
	parent.__init(self)

    self.iH = iH
    self.iW = iW
    self.oH = oH
    self.oW = oW
    self.kernelRows = interpKernel(iH, oH)
    self.kernelCols = interpKernel(iW, oW)
    self.kernelRowsT = self.kernelRows:t():clone()
    self.kernelColsT = self.kernelCols:t():clone()
    self.buffer = torch.Tensor(oH, oW)
    self.buffer2 = torch.Tensor(iH,oW)

    if false then
	self.hermitianRows = hermitianRows or false
	self.hermitianCols = hermitianCols or false
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
end
	

-- assuming inputs are size [dim1 x dim2 x nRows x nCols x 2]
-- or [dim1 x nRows x nCols x 2]
function ComplexInterp:updateOutput(input)
   if input:type() ~= 'torch.CudaTensor' then
      interpolateBatch(self.iH, self.iW, self.oH, self.oW, input, self.output, self.kernelRows, self.kernelCols)
  else
     self.buffer:resize(self.oH, self.iW)
     if input:nDimension() == 4 then
        self.output:resize(input:size(1), self.oH, self.oW, 2)
     elseif input:nDimension() == 5 then 
        self.output:resize(input:size(1), input:size(2), self.oH, self.oW, 2)
     else
        error('invalid input size')
     end
     cucomplex.complexInterp_interpolate(input, self.output, self.kernelRows, self.kernelCols, self.buffer)
  end
	return self.output
end


function ComplexInterp:updateGradInput(input, gradOutput)
   if input:type() ~= 'torch.CudaTensor' then
	local ndim = gradOutput:nDimension()
	local size = gradOutput:size()
	if ndim == 5 then
		gradOutput:resize(size[1]*size[2],size[3],size[4],2)
	end
	local nFrames = gradOutput:size(1)
	self.gradInput:resize(nFrames, self.iH, self.iW, 2)
	self.gradInput:zero()
	for i = 1,nFrames do
		self.buffer2:zero()
		-- downsample real part
		self.buffer2:addmm(self.kernelRows,gradOutput[i][{{},{},1}])
		self.gradInput[i][{{},{},1}]:addmm(self.buffer2,self.kernelCols:t())
		-- downsample imaginary part
		self.buffer2:zero()
		self.buffer2:addmm(self.kernelRows,gradOutput[i][{{},{},2}])
		self.gradInput[i][{{},{},2}]:addmm(self.buffer2,self.kernelCols:t())
	end
	if ndim == 5 then
		gradOutput:resize(size)
		self.gradInput:resize(size[1],size[2],self.iH,self.iW,2)
	end
	--print('complex interp gradInput norm = ',self.gradInput:norm())

    else
     if gradOutput:nDimension() == 4 then
        self.gradInput:resize(gradOutput:size(1), self.iH, self.iW, 2)
     elseif gradOutput:nDimension() == 5 then 
        self.gradInput:resize(gradOutput:size(1), gradOutput:size(2), self.iH, self.iW, 2)
     else
        error('invalid gradOutput size')
     end
     
     cucomplex.complexInterp_interpolate(gradOutput, self.gradInput, self.kernelRows, self.kernelCols, self.buffer)
  end
	return self.gradInput
end

-- only use this on CPU
function interpolateBatch(iH, iW, oH, oW, input, output, kernelRows, kernelCols)
	local ndim = input:nDimension()
	local size = input:size()
	if ndim == 5 then
		-- need this hack since reshape module only supports 4 dimensions
		input:resize(size[1]*size[2], iH, iW, 2)
	end
	local nFrames = input:size(1)
    
	output:resize(nFrames, oH, oW, 2)	
	output:zero()
	for i = 1,nFrames do
       local real = input[i]:select(3,1)
       local imag = input[i]:select(3,2)
       output[i][{{},{},1}]:copy((kernelRows:t()*real)*kernelCols)
       output[i][{{},{},2}]:copy((kernelRows:t()*imag)*kernelCols)
	end	
	if ndim == 5 then
		input:resize(size)
		output:resize(size[1],size[2],oH,oW,2)
     end
end

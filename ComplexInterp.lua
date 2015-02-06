--module to perform complex interpolation on inputs. 
--real and imaginary parts are interpolated separately

require 'nn'
require 'interp'

local ComplexInterp, parent = torch.class('nn.ComplexInterp','nn.Module')

function ComplexInterp:__init(iH, iW, oH, oW, interpType)
	parent.__init(self)
    self.iH = iH
    self.iW = iW
    self.oH = oH
    self.oW = oW
    self.kernelRows = interpKernel(iH, oH, interpType)
    self.kernelCols = interpKernel(iW, oW, interpType)
    --self.kernelRows:mul(2)
    --self.kernelCols:mul(2)
    self.gradInput = torch.Tensor(1,1,iH,iW,2) 
end
	
-- assuming inputs are size [dim1 x dim2 x nRows x nCols x 2]
-- or [dim1 x nRows x nCols x 2]
function ComplexInterp:updateOutput(input)
   if input:type() ~= 'torch.CudaTensor' or global_debug1 then
      interpolateBatch(self.iH, self.iW, self.oH, self.oW, input, self.output, self.kernelRows, self.kernelCols)
  else
     if input:nDimension() == 4 then
        self.output:resize(input:size(1), self.oH, self.oW, 2)
     elseif input:nDimension() == 5 then 
        self.output:resize(input:size(1), input:size(2), self.oH, self.oW, 2)
     else
        error('invalid input size')
     end
     spectralcuda.complexInterp_interpolate(input, self.output, self.kernelRows, self.kernelCols)
  end
  -- only keep real part
  self.output:select(input:nDimension(),2):zero()
  return self.output
end


function ComplexInterp:updateGradInput(input, gradOutput)
   if input:type() ~= 'torch.CudaTensor' or global_debug2 then
      interpolateBatch(self.oH, self.oW, self.iH, self.iW, gradOutput, self.gradInput, self.kernelRows:t(), self.kernelCols:t())
   else
      if gradOutput:nDimension() == 4 then
         self.gradInput:resize(gradOutput:size(1), self.iH, self.iW, 2)
      elseif gradOutput:nDimension() == 5 then 
         self.gradInput:resize(gradOutput:size(1), gradOutput:size(2), self.iH, self.iW, 2)
      else
         error('invalid gradOutput size')
      end      
      spectralcuda.complexInterp_interpolate(gradOutput, self.gradInput, self.kernelRows, self.kernelCols)
   end
   -- only keep real part
   self.gradInput:select(input:nDimension(),2):zero()
   return self.gradInput
end

-- this works on CPU or GPU, but is slow on GPU
function interpolateBatch(iH, iW, oH, oW, input, output, kernelRows, kernelCols)
	local ndim = input:nDimension()
	local size = input:size()
	if ndim == 5 then
		input:resize(size[1]*size[2], iH, iW, 2)
	end
	local nFrames = input:size(1)    
	output:resize(nFrames, oH, oW, 2)	
	output:zero()
    local buffer
    if input:type() ~= 'torch.CudaTensor' then
       buffer = torch.FloatTensor(oH,iW)
    else 
       buffer = torch.CudaTensor(oH,iW)
    end
	for i = 1,nFrames do
       local real = input[i]:select(3,1)
       local imag = input[i]:select(3,2)
       buffer:zero()
       buffer:addmm(kernelRows:t(),real)
       output[i][{{},{},1}]:zero()
       output[i][{{},{},1}]:addmm(buffer,kernelCols)
       buffer:zero()
       buffer:addmm(kernelRows:t(),imag)
       output[i][{{},{},2}]:zero()
       output[i][{{},{},2}]:addmm(buffer,kernelCols)
	end	
	if ndim == 5 then
		input:resize(size)
		output:resize(size[1],size[2],oH,oW,2)
     end
end

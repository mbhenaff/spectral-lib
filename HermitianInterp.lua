--module to perform complex interpolation on inputs. 
--real and imaginary parts are interpolated separately
--optionally we can choose the outputs to be hermitian, i.e. elements 1 and N/2+1 are real. 
--this assumes the input already represents only half of the elements (the others are assumed to be their complex conjugates)

require 'nn'
require 'interp'

local HermitianInterp, parent = torch.class('nn.HermitianInterp','nn.Module')

-- inputSize should be a tensor or table {nRows,nCols}, same for outputSize 
-- hermitianRows/hermitianCols should be true/false indicating if we assume the inputs are hermitian

function HermitianInterp:__init(sH,sW,iH,iW)
	parent.__init(self)
	self.sH = sH
	self.sW = sW
    self.iH = iH
    self.iW = iW
	self.kernelRows = interpKernel(sH,iH)
	self.kernelCols = interpKernel(sW,iW)
    -- make symmetrizer matrix
    self.S = torch.zeros(sH,sH)
    for i = 1,sH do
       self.S[i][i] = 0.5
       self.S[i][sH-i+1] = 0.5
    end
    self.pad = torch.Tensor(sH,sW,2)
	self.buffer = torch.Tensor(iH,sW)
	self.buffer2 = torch.Tensor(sH,iW)
end
	

-- assuming inputs are size [dim1 x dim2 x nRows x nCols x 2]
-- or [dim1 x nRows x nCols x 2]
function HermitianInterp:updateOutput(input)
   local ndim = input:nDimension()
   local size = input:size()
   if ndim == 5 then
      -- need this hack since reshape module only supports 4 dimensions
      input:resize(size[1]*size[2],size[3],size[4],2)
   end
   local nFrames = input:size(1)
   self.output:resize(nFrames, self.iH, self.iW, 2)	
   self.output:zero()
   for i = 1,nFrames do
      self.pad:copy(input[i])
      -- first and last columns real and symmetric
      self.pad[{{},1,1}]:zero()
      self.pad[{{},1,1}]:addmv(self.S,input[i][{{},1,1}])
      self.pad[{{},self.sW,1}]:zero()
      self.pad[{{},self.sW,1}]:addmv(self.S,input[i][{{},self.sW,1}])
      self.pad[{{},1,2}]:zero()
      self.pad[{{},self.sW,2}]:zero()
      -- interpolate real part
      self.buffer:zero()
      self.buffer:addmm(self.kernelRows:t(),self.pad[{{},{},1}])
      self.output[i][{{},{},1}]:addmm(self.buffer,self.kernelCols)
      -- interpolate imaginary part
      self.buffer:zero()
      self.buffer:addmm(self.kernelRows:t(),self.pad[{{},{},2}])
      self.output[i][{{},{},2}]:addmm(self.buffer,self.kernelCols)
   end	
   if ndim == 5 then
      input:resize(size)
      self.output:resize(size[1],size[2],self.iH,self.iW,2)
   end
   return self.output
end



function HermitianInterp:updateGradInput(input, gradOutput)
	local ndim = gradOutput:nDimension()
	local size = gradOutput:size()
	if ndim == 5 then
		gradOutput:resize(size[1]*size[2],size[3],size[4],2)
	end
	local nFrames = gradOutput:size(1)
	self.gradInput:resize(nFrames, self.sH, self.sW, 2)
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
        -- make first and last columns real and symmetric
        self.pad:copy(self.gradInput[i])
        self.gradInput[i][{{},1,2}]:zero()
        self.gradInput[i][{{},self.sW,2}]:zero()
        self.gradInput[i][{{},1,1}]:zero()
        self.gradInput[i][{{},self.sW,1}]:zero()
        self.gradInput[i][{{},1,1}]:addmv(self.S,self.pad[{{},1,1}])
        self.gradInput[i][{{},self.sW,1}]:addmv(self.S,self.pad[{{},self.sW,1}])
	end
	if ndim == 5 then
		gradOutput:resize(size)
		self.gradInput:resize(size[1],size[2],self.sH,self.sW,2)
	end
	return self.gradInput
end
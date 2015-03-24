
local SpectralConvolution, parent = torch.class('nn.SpectralConvolution','nn.Module')

function SpectralConvolution:__init(batchSize, nInputMaps, nOutputMaps, dim, subdim, GFTMatrix)
   parent.__init(self)
   self.dim = dim
   self.subdim = subdim
   self.nInputMaps = nInputMaps
   self.nOutputMaps = nOutputMaps
   self.GFTMatrix = GFTMatrix or torch.eye(dim,dim)
   self.iGFTMatrix = self.GFTMatrix:t():clone()
   self.interpType = 'bilinear'
   -- bias
   self.bias = torch.Tensor(nOutputMaps)
   self.gradBias = torch.Tensor(nOutputMaps)
   -- buffers in spectral domain (TODO: use global buffer)
   self.inputSpectral = torch.Tensor(batchSize, nInputMaps, dim)
   self.outputSpectral = torch.Tensor(batchSize, nOutputMaps, dim)
   self.gradInputSpectral = torch.Tensor(batchSize, nInputMaps, dim)
   self.gradOutputSpectral = torch.Tensor(batchSize, nOutputMaps, dim)
   self.output = torch.Tensor(batchSize, nOutputMaps, dim)
   self.weight = torch.Tensor(nOutputMaps, nInputMaps, dim)
   self.gradWeight = torch.Tensor(nOutputMaps, nInputMaps, dim)

   -- weight transformation (interpolation)
   local weightTransform = nn.Interp(self.subdim, self.dim, self.interpType)
   self:setWeightTransform(weightTransform, torch.LongStorage({nOutputMaps, nInputMaps, self.subdim}))
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   self:reset()
end

function SpectralConvolution:reset(stdv)
   stdv = 1/math.sqrt(self.nInputMaps*self.subdim)
   self.bias:uniform(-stdv, stdv)
   self.weightPreimage:uniform(-stdv,stdv)
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
end

-- use this after sending to GPU
function SpectralConvolution:resetPointers()
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
end


-- apply graph fourier transform on the input, store result in output
function SpectralConvolution:batchGFT(input, output, dir) 
   local b = input:size(1)
   local f = input:size(2)
   local d = input:size(3)
   input:resize(b*f, d)
   output:resize(b*f, d)
   output:zero()
   if dir == 1 then
      output:addmm(input, self.GFTMatrix)
   elseif dir == -1 then
      output:addmm(input, self.iGFTMatrix)
   else
      error('dir should be 1 or -1')
   end
   input:resize(b, f, d)
   output:resize(b, f, d)
end

function SpectralConvolution:updateOutput(input)
   -- forward GFT
   self:batchGFT(input, self.inputSpectral, 1)
   -- product in spectral domain
   libspectralnet.prod_fprop_real(self.inputSpectral, self.weight, self.outputSpectral) 
   -- inverse GFT
   self:batchGFT(self.outputSpectral, self.output, -1)
   -- add bias
   self.output:resize(self.output:size(1), self.output:size(2), self.output:size(3), 1)
   libspectralnet.bias_updateOutput(self.bias, self.output)
   self.output:resize(self.output:size(1), self.output:size(2), self.output:size(3))
   return self.output
end
   
function SpectralConvolution:updateGradInput(input, gradOutput)
   -- forward GFT
   self:batchGFT(gradOutput, self.gradOutputSpectral, 1)
   -- product 
   libspectralnet.prod_bprop_real(self.gradOutputSpectral, self.weight, self.gradInputSpectral)
   -- inverse GFT
   self:batchGFT(self.gradInputSpectral, self.gradInput, -1) 
   return self.gradInput
end

function SpectralConvolution:accGradParameters(inputs, gradOutput, scale)
   local scale = scale or 1
   libspectralnet.prod_accgrad_real(self.inputSpectral, self.gradOutputSpectral, self.gradWeight)
   gradOutput:resize(gradOutput:size(1), gradOutput:size(2), gradOutput:size(3), 1)
   libspectralnet.bias_accGradParameters(self.gradBias, gradOutput, scale)
   gradOutput:resize(gradOutput:size(1), gradOutput:size(2), gradOutput:size(3))
end

  
function SpectralConvolution:printNorms()
   print('-------------------')
   print('weightPreimage norm = ' .. self.weightPreimage:norm())
   print('gradWeightPreimage norm = ' .. self.gradWeightPreimage:norm())
   print('weight norm = ' .. self.weight:norm())
   print('gradWeight norm = ' .. self.gradWeight:norm())
   print('-------------------')
end

      
      

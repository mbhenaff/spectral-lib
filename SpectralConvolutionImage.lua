require 'interp'
require 'complex' 
require 'image'
require 'Interp'
require 'HermitianInterp'
require 'libFFTconv'
local cufft = dofile('cufft/cufft.lua')

-- Module for performing convolution in the frequency domain. 
-- interpType refers to the type of interpolation kernel we use on the subsampled weights,
-- to make them as large as the input image.
-- realKernels specifies whether we want our kernels to be real (in the frequency domain)
local SpectralConvolutionImage, parent = torch.class('nn.SpectralConvolutionImage','nn.Module')

function SpectralConvolutionImage:__init(batchSize, nInputPlanes, nOutputPlanes, iH, iW, sH, sW, interpType,realKernels)
   parent.__init(self)

   if not (iW % 2 == 0 and iH % 2 == 0) then
      error('input width should be even. Best if a power of 2. Now iW=' .. iW .. ', iH=' .. iH)
   end
   self.interpType = interpType or 'bilinear'
   self.batchSize = batchSize
   self.nInputPlanes = nInputPlanes
   self.nOutputPlanes = nOutputPlanes
   self.realKernels = realKernels or true
   -- width/height of inputs
   self.iW = iW
   self.iH = iH
   -- width/height of subsampled weights
   self.sW = sW
   self.sH = sH
   -- weight transformation
   self.gradInput = torch.Tensor(batchSize, nInputPlanes, iH, iW)
   self.gradWeight = torch.Tensor(nOutputPlanes, nInputPlanes, iH, iW, 2)
   local weightTransform = nn.InterpImage(sH,sW,iH,iW,self.interpType)
   self:setWeightTransform(weightTransform,torch.LongStorage({nOutputPlanes,nInputPlanes,sH,sW,2}))
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   self:reset()
end

function SpectralConvolutionImage:reset(stdv)
   stdv = 1/math.sqrt(self.nInputPlanes*self.sW*self.sH)
   self.weightPreimage:uniform(-stdv,stdv)
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   if self.realKernels then
      self.weightPreimage:select(5,2):zero()
   end
end

function SpectralConvolutionImage:updateOutput(input) 
   -- initialize buffers
   self.inputSpectral = global_buffer1
   self.outputSpectral = global_buffer2
   self.outputSpectral:resize(self.batchSize, self.nOutputPlanes, self.iH, self.iW, 2)
   self.inputSpectral:resize(self.batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.output:resize(self.outputSpectral:size())
   -- forward FFT
   self.inputSpectral:zero()
   self.inputSpectral:select(5,1):copy(input)
   cufft.fft2d_c2c(self.inputSpectral,self.inputSpectral,1)
   -- product
   libFFTconv.prod_fprop(self.inputSpectral,self.weight,self.outputSpectral,true)
   -- inverse FFT
   cufft.fft2d_c2c(self.outputSpectral,self.output,-1)
   return self.output
end

-- note that here gradOutput is the same size as input
function SpectralConvolutionImage:updateGradInput(input, gradOutput)
   -- initialize buffers
   self.gradInputSpectral = global_buffer1
   self.gradOutputSpectral = global_buffer2
   self.gradInputSpectral:resize(self.batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.gradOutputSpectral:resize(self.batchSize, self.nOutputPlanes, self.iH, self.iW,2)
   -- forward FFT
   cufft.fft2d_c2c(gradOutput,self.gradOutputSpectral,1)
   -- product
   libFFTconv.prod_bprop(self.gradOutputSpectral, self.weight, self.gradInputSpectral,false)
   -- inverse FFT
   cufft.fft2d_c2c(self.gradInputSpectral,self.gradInputSpectral,-1)
   self.gradInput:copy(self.gradInputSpectral:select(5,1))
   return self.gradInput
end

function SpectralConvolutionImage:accGradParameters(input, gradOutput, scale)
   scale  = scale or 1
   self.inputSpectral = global_buffer1
   self.gradOutputSpectral = global_buffer2
   self.inputSpectral:resize(self.batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.gradOutputSpectral:resize(self.batchSize, self.nOutputPlanes, self.iH, self.iW,2)
   -- forward FFT
   cufft.fft2d_c2c(gradOutput,self.gradOutputSpectral,1)
   self.inputSpectral:zero()
   self.inputSpectral:select(5,1):copy(input)
   cufft.fft2d_c2c(self.inputSpectral,self.inputSpectral,1)
   -- product
   libFFTconv.prod_accgrad(self.inputSpectral,self.gradOutputSpectral,self.gradWeight,true)
   self.gradWeight:div(self.iW * self.iH)
   cutorch.synchronize()
end

-------------------------------------
-- DEBUG
-------------------------------------

-- apply inverse FFT to the weights and return the real/imaginary parts in the spatial domain
-- as well the complex modulus in the frequency domain. 
function SpectralConvolutionImage:printFilters()
   local spatial_real = {}
   local spatial_imag = {}
   local freq_mod = {}
   local spatialFilters = torch.CudaTensor(self.weight:size())
   cufft.fft2d_c2c(self.weight,spatialFilters,-1)
   for i = 1,self.nOutputPlanes do
      for j = 1,self.nInputPlanes do
         local mod = self.weight[i][j]:norm(2,3)
         table.insert(freq_mod,mod:float():squeeze())
         table.insert(spatial_real,reshapeFilter(spatialFilters[i][j]:select(3,1)))
         table.insert(spatial_imag,reshapeFilter(spatialFilters[i][j]:select(3,2)))
      end
   end
   return spatial_real,spatial_imag,freq_mod
end

function printFilters2(model)
   local spatial_real = torch.CudaTensor(model.nOutputPlanes, model.nInputPlanes, model.iH, model.iW)
   local spatial_imag = torch.CudaTensor(model.nOutputPlanes, model.nInputPlanes, model.iH, model.iW)
   local freq_mod = torch.CudaTensor(model.nOutputPlanes, model.nInputPlanes, model.iH, model.iW)  
   local spatialFilters = torch.CudaTensor(model.weight:size())
   cufft.fft2d_c2c(model.weight,spatialFilters,-1)
   for i = 1,model.nOutputPlanes do
      for j = 1,model.nInputPlanes do
         local mod = model.weight[i][j]:norm(2,3)
         freq_mod[i][j]:copy(mod)
         spatial_real[i][j]:copy(reshapeFilter(spatialFilters[i][j]:select(3,1)))
         spatial_imag[i][j]:copy(reshapeFilter(spatialFilters[i][j]:select(3,2)))
      end
   end
   return spatial_real,spatial_imag,freq_mod
end


function isnan(x)
   return x ~= x
end

-- reshape filter to that its center frequency is in the middle of the image 
-- instead of at the corners.
function reshapeFilter(x)
   local r = x:size(1)
   local c = x:size(2)
   local y = torch.Tensor(r,c)
   y[{{1,r/2},{1,c/2}}]:copy(x[{{r/2+1,r},{c/2+1,c}}])
   y[{{r/2+1,r},{c/2+1,c}}]:copy(x[{{1,r/2},{1,c/2}}])
   y[{{1,r/2},{c/2+1,c}}]:copy(x[{{r/2+1,r},{1,c/2}}])
   y[{{r/2+1,r},{1,c/2}}]:copy(x[{{1,r/2},{c/2+1,c}}])
   return y
end
   
function SpectralConvolutionImage:printNorms()
   print('-------------------')
   print('weightPreimage norm = ' .. self.weightPreimage:norm())
   print('gradWeightPreimage norm = ' .. self.gradWeightPreimage:norm())
   print('weight norm = ' .. self.weight:norm())
   print('gradWeight norm = ' .. self.gradWeight:norm())
   print('-------------------')
end

function SpectralConvolutionImage:checkForErrors()
   if isnan(self.output:norm()) then
      error('self.output has nan')
   end
   if isnan(self.gradInput:norm()) then
      error('self.gradInput has nan')
   end
   if isnan(self.gradBias:norm()) then
      error('self.gradBias has nan')
   end
end



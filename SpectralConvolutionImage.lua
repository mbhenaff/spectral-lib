
-- Module for performing convolution in the frequency domain. 
-- interpType refers to the type of interpolation kernel we use on the subsampled weights,
-- to make them as large as the input image.
-- realKernels specifies whether we want our kernels to be real (in the frequency domain)
local SpectralConvolutionImage, parent = torch.class('nn.SpectralConvolutionImage','nn.Module')

function SpectralConvolutionImage:__init(nInputPlanes, nOutputPlanes, iH, iW, sH, sW, interpType, real)
   parent.__init(self)
   assert(iW % 2 == 0 and iH % 2 == 0, 'input size should be even')
   assert(sH % 2 == 1 and sW % 2 == 1, 'kernel size should be odd')

   self.interpType = interpType or 'bilinear'
   self.nInputPlanes = nInputPlanes
   self.nOutputPlanes = nOutputPlanes
   self.makeReal = real or 'realpart'
   -- width/height of images
   self.iW = iW
   self.iH = iH
   -- width/height of subsampled weights
   self.sW = sW
   self.sH = sH
   -- how many rows/cols on borders to zero out
   self.zW = (sW-1)/2
   self.zH = (sH-1)/2
   -- bias 
   if self.makeReal == 'realpart' then
      self.bias = torch.Tensor(nOutputPlanes)
      self.gradBias = torch.Tensor(nOutputPlanes)
   end
   -- make buffers to store spectral representations 
   global_buffer1 = global_buffer1 or torch.CudaTensor()
   global_buffer2 = global_buffer2 or torch.CudaTensor()
   global_buffer3 = global_buffer3 or torch.CudaTensor()
   -- weight transformation
   self.gradInput = torch.Tensor()
   self.gradWeight = torch.Tensor()
   local weightTransform = nn.InterpImage(sH,sW,iH,iW,self.interpType)
   self:setWeightTransform(weightTransform,torch.LongStorage({nOutputPlanes,nInputPlanes,sH,sW,2}))
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   self:reset()
end

function SpectralConvolutionImage:reset(stdv)
   local stdv = stdv or 1/math.sqrt(self.nInputPlanes*self.sW*self.sH)
   if self.makeReal == 'realpart' then
      self.bias:uniform(-stdv,stdv)
   end
   self.weightPreimage:uniform(-stdv,stdv)
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   if self.makeReal == 'realpart' then
      self.weightPreimage:select(5,2):zero()
   end
end


function SpectralConvolutionImage:updateOutput(input) 
   -- initialize buffers
   local batchSize = input:size(1)
   self.inputSpectral = global_buffer1
   self.outputSpectral = global_buffer2
   self.inputSpectral:resize(batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.outputSpectral:resize(batchSize, self.nOutputPlanes, self.iH, self.iW, 2)

   -- forward FFT
   self.inputSpectral:zero()
   self.inputSpectral:select(5,1):copy(input)
   cufft.fft2d_c2c(self.inputSpectral,self.inputSpectral,1)
   -- product
   libspectralnet.prod_fprop_complex(self.inputSpectral,self.weight,self.outputSpectral,true)
   -- inverse FFT
   cufft.fft2d_c2c(self.outputSpectral,self.outputSpectral,-1)

   -- make output real
   if self.makeReal == 'realpart' then
      self.output:resize(batchSize, self.nOutputPlanes, self.iH, self.iW)
      self.output:copy(self.outputSpectral:select(5,1))
      -- add bias
      libspectralnet.bias_updateOutput(self.bias, self.output)
      -- zero borders
      libspectralnet.crop_zeroborders(self.output, self.zH, self.zW)
   else
      self.output:resize(batchSize, self.nOutputPlanes, self.iH, self.iW,2)
      self.output:copy(self.outputSpectral)
   end

   return self.output
end

-- note that here gradOutput is the same size as input
function SpectralConvolutionImage:updateGradInput(input, gradOutput)
   -- initialize buffers
   local batchSize = input:size(1)
   self.gradInputSpectral = global_buffer1
   self.gradOutputSpectral = global_buffer2
   self.gradInputSpectral:resize(batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.gradOutputSpectral:resize(batchSize, self.nOutputPlanes, self.iH, self.iW,2)
   self.gradInput:resize(batchSize, self.nInputPlanes, self.iH, self.iW)

   if self.makeReal == 'realpart' then 
      self.gradOutputCropped = global_buffer3
      self.gradOutputCropped:resize(batchSize, self.nOutputPlanes, self.iH, self.iW)
      self.gradOutputCropped:copy(gradOutput)
      -- zero borders
      libspectralnet.crop_zeroborders(self.gradOutputCropped, self.zH, self.zW)
      -- make complex
      self.gradOutputSpectral:select(5,1):copy(self.gradOutputCropped)
      self.gradOutputSpectral:select(5,2):zero()
   else
      self.gradOutputSpectral:copy(gradOutput)
   end

   -- forward FFT
   cufft.fft2d_c2c(self.gradOutputSpectral,self.gradOutputSpectral,1)
   -- product
   libspectralnet.prod_bprop_complex(self.gradOutputSpectral, self.weight, self.gradInputSpectral,false)
   -- inverse FFT
   cufft.fft2d_c2c(self.gradInputSpectral,self.gradInputSpectral,-1)
   self.gradInput:copy(self.gradInputSpectral:select(5,1))
   return self.gradInput
end

function SpectralConvolutionImage:accGradParameters(input, gradOutput, scale)
   scale  = scale or 1
   self.gradWeight:zero()
   -- initialize buffers
   local batchSize = input:size(1)
   self.inputSpectral = global_buffer1
   self.gradOutputSpectral = global_buffer2
   self.gradOutputCropped = global_buffer3
   self.inputSpectral:resize(batchSize, self.nInputPlanes, self.iH, self.iW, 2)
   self.gradOutputSpectral:resize(batchSize, self.nOutputPlanes, self.iH, self.iW,2)
   -- make gradOutput complex
   if self.makeReal == 'realpart' then 
      -- zero borders 
      self.gradOutputCropped:resize(batchSize, self.nOutputPlanes, self.iH, self.iW)
      self.gradOutputCropped:copy(gradOutput)
      libspectralnet.crop_zeroborders(self.gradOutputCropped, self.zH, self.zW)
      self.gradOutputSpectral:select(5,1):copy(self.gradOutputCropped)
      self.gradOutputSpectral:select(5,2):zero()
   else
      self.gradOutputSpectral:copy(gradOutput)
   end
   cufft.fft2d_c2c(self.gradOutputSpectral,self.gradOutputSpectral,1)

   -- forward FFT
   self.inputSpectral:zero()
   self.inputSpectral:select(5,1):copy(input)
   cufft.fft2d_c2c(self.inputSpectral,self.inputSpectral,1)
   -- product
   libspectralnet.prod_accgrad_complex(self.inputSpectral,self.gradOutputSpectral,self.gradWeight,true)
   self.gradWeight:div(self.iW * self.iH)
   
   if self.bias then
      -- bias gradient
      libspectralnet.bias_accGradParameters(self.gradBias, gradOutput, scale)
   end
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
   libspectralnet.fft2d_c2c(self.weight,spatialFilters,-1)
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
   libspectralnet.fft2d_c2c(model.weight,spatialFilters,-1)
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



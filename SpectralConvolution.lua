require 'interp'
require 'complex' 
require 'image'
require 'HermitianInterp'

local cufft = dofile('cufft/cufft.lua')

local SpectralConvolution, parent = torch.class('nn.SpectralConvolution','nn.Module')

function SpectralConvolution:__init(batchSize, nInputPlanes, nOutputPlanes, iH, iW, sH, sW, fullweights)
   parent.__init(self)

   if not (iW % 2 == 0 and iH % 2 == 0) then
      error('input width should be even. Best if a power of 2')
   end
   -- set to false for hermitian weights, true for full set of weights
   self.fullweights = fullweights or false

   self.batchSize = batchSize
   self.nInputPlanes = nInputPlanes
   self.nOutputPlanes = nOutputPlanes

   -- width/height of inputs
   self.iW = iW
   self.iH = iH
   -- width/height of subsampled weights
   self.sW = sW
   self.sH = sH

   -- representations in original domain
   self.output = torch.Tensor(batchSize, nOutputPlanes, iH, iW)
   self.gradInput = torch.Tensor(batchSize, nInputPlanes, iH, iW)
   self.bias = torch.Tensor(nOutputPlanes)
   self.gradBias = torch.Tensor(nOutputPlanes)
   -- buffers in spectral domain (TODO: use single global buffer)
   if self.fullweights then
      self.inputSpectral = torch.Tensor(batchSize, nInputPlanes, iH, iW, 2)
      self.outputSpectral = torch.Tensor(batchSize, nOutputPlanes, iH, iW, 2)
      self.gradInputSpectral = torch.Tensor(batchSize, nInputPlanes, iH, iW, 2)
      self.gradWeight = torch.Tensor(nOutputPlanes, nInputPlanes, iH, iW, 2)
      self.gradOutputSpectral = torch.Tensor(batchSize, nOutputPlanes, iH, iW,2)	
   else
      self.inputSpectral = torch.Tensor(batchSize, nInputPlanes, iH, iW/2+1, 2)
      self.outputSpectral = torch.Tensor(batchSize, nOutputPlanes, iH, iW/2+1, 2)
      self.gradInputSpectral = torch.Tensor(batchSize, nInputPlanes, iH, iW/2+1, 2)
      self.gradOutputSpectral = torch.Tensor(batchSize, nOutputPlanes, iH, iW/2+1,2)	
      self.gradWeight = torch.Tensor(nOutputPlanes, nInputPlanes, iH, iW/2+1, 2)
   end

   -- weight transformation
   local weightTransform
   if self.fullweights then
      weightTransform = nn.ComplexInterp({sH,sW},{iH,iW},false,false)
   else
      weightTransform = nn.HermitianInterp(sH,sW,iH,iW/2+1)
   end
   self:setWeightTransform(weightTransform,torch.LongStorage({nOutputPlanes,nInputPlanes,sH,sW,2}))
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   self:reset()
end


function SpectralConvolution:reset(stdv)
   -- TODO: find appropriate initialization range
   stdv = 1/math.sqrt(self.nInputPlanes)
   self.weightPreimage:uniform(-stdv,stdv)
   self.weight = self.transformWeight:updateOutput(self.weightPreimage)
   self.gradWeightPreimage = self.transformWeight:updateGradInput(self.weightPreimage,self.gradWeight)
   self.bias:uniform(-stdv,stdv)
end

function SpectralConvolution:updateOutput(input,flag) 
   -- forward FFT
   if self.fullweights then
      self.inputSpectral:zero()
      self.inputSpectral:select(5,1):copy(input)
      cufft.fft2d_c2c(self.inputSpectral,self.inputSpectral,1,true)
   else
      cufft.fft2d_r2c(input,self.inputSpectral) 
   end
   -- compute product in spectral domain
   complex.conj(self.weight)
   cucomplex.prod_fprop(self.inputSpectral,self.weight,self.outputSpectral)
   --complex.prod_fprop(self.inputSpectral,self.weight,self.outputSpectral)
   complex.conj(self.weight)
   -- inverse FFT
   if self.fullweights then
      cufft.fft2d_c2c(self.outputSpectral,self.outputSpectral,-1,true)
      self.output:copy(self.outputSpectral:select(5,1))
   else
      cufft.fft2d_c2r(self.outputSpectral, self.output)
   end
   return self.output
end

-- note that here gradOutput is the same size as input
function SpectralConvolution:updateGradInput(input, gradOutput,flag)
   -- forward FFT
   if self.fullweights then
      self.gradOutputSpectral:zero()
      self.gradOutputSpectral:select(5,1):copy(gradOutput)
      cufft.fft2d_c2c(self.gradOutputSpectral,self.gradOutputSpectral,1,false)
   else
      cufft.fft2d_r2c(gradOutput, self.gradOutputSpectral)   
   end
   -- product in spectral domain
   self.gradInputSpectral:zero()
   cucomplex.prod_bprop(self.gradOutputSpectral, self.weight, self.gradInputSpectral)
   -- inverse FFT
   if self.fullweights then
      cufft.fft2d_c2c(self.gradInputSpectral,self.gradInputSpectral,-1,false)
      self.gradInput = self.gradInputSpectral:select(5,1)
   else 
      cufft.fft2d_c2r(self.gradInputSpectral, self.gradInput)
   end
   return self.gradInput
end

function SpectralConvolution:accGradParameters(input, gradOutput, scale)
   scale  = scale or 1
   -- forward FFT
   if self.fullweights then
      self.gradOutputSpectral:zero()
      self.gradOutputSpectral:select(5,1):copy(gradOutput)
      cufft.fft2d_c2c(self.gradOutputSpectral,self.gradOutputSpectral,1,false)
   else
      cufft.fft2d_r2c(gradOutput, self.gradOutputSpectral)   
   end
   -- product in spectral domain
   complex.conj(self.gradOutputSpectral)
   cucomplex.prod_accgrad(self.inputSpectral,self.gradOutputSpectral,self.gradWeight)
   complex.conj(self.gradOutputSpectral)
   -- normalize
   self.gradWeight:div(self.iW * self.iH)
   if not self.fullweights then
      -- hack to pass the unit tests (forward jacobian gives certain coefficients twice the influence due to hermitian symmetries)
      self.gradWeight[{{},{},{},{2,self.iW/2},{}}]:mul(2)
   end
end


-------------------------------------
-- DEBUG
-------------------------------------

function SpectralConvolution:printFilters()
   local imgs = {}
   for i = 1,self.nOutputPlanes do
      for j = 1,self.nInputPlanes do
         -- compute modulus
         local mod = self.weight[i][j]:norm(2,3)
         table.insert(imgs,mod:double():squeeze())
         -- compute iFFT
         cufft.fft1d_c2r(self.weight[i][j],self.tmp)
         table.insert(imgs,self.GFTMatrix:clone():double() * self.tmp:clone():double())
      end
   end
   for i=1,#imgs do
      gnuplot.figure(i)
      gnuplot.imagesc(imgs[i])
   end
end


function isnan(x)
   return x ~= x
end

function SpectralConvolution:printNorms()
   print('-------------------')
   print('weightPreimage norm = ' .. self.weightPreimage:norm())
   print('gradWeightPreimage norm = ' .. self.gradWeightPreimage:norm())
   print('weight norm = ' .. self.weight:norm())
   print('bias norm = ' .. self.bias:norm())
   print('gradWeight norm = ' .. self.gradWeight:norm())
   print('gradBias norm = ' .. self.gradBias:norm())
   print('-------------------')
end

function SpectralConvolution:checkForErrors()
   if isnan(self.output:norm()) then
      error('self.output has nan')
   end
   if isnan(self.bias) then
      error('self.bias has nan')
   end
   if isnan(self.gradInput:norm()) then
      error('self.gradInput has nan')
   end
   if isnan(self.gradBias:norm()) then
      error('self.gradBias has nan')
   end
end



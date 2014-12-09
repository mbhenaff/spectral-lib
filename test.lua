-- Unit tests and speed tests for all the modules. 

require 'cunn'
require 'cucomplex'
require 'HermitianInterp'
require 'Interp'
require 'ComplexInterp'
require 'Real'
require 'Bias'
require 'Crop'
require 'SpectralConvolutionImage'
require 'SpectralConvolution'
require 'Jacobian2'
require 'utils'
cufft = dofile('cufft/cufft.lua')

torch.manualSeed(123)
cutorch.setDevice(4)

local test_correctness = true
local test_crop = false
local test_bias = false
local test_interp = true
local test_real = false
local test_complex_interp = false
local test_spectralconv_img = false
local test_spectralconv = true
local test_time = true

local mytester = torch.Tester()
local jac = nn.Jacobian
local sjac
local nntest = {}
local precision = 1e-1

if test_crop then 
   function nntest.Crop()
      print('\n')
      local iW = 8
      local iH = 8
      local rows = 2
      local cols = 2
      local batchSize = 1
      local nPlanes = 1
      model = nn.Crop(iH,iW,rows,cols,false)
      model = model:cuda()
      input = torch.CudaTensor(batchSize,nPlanes,iH,iW)
      err,jf,jb = jac.testJacobian(model, input)
      print('error on state = ' .. err)
      mytester:assertlt(err,precision, 'error on crop')
      print('\n')
   end
end

if test_bias then
   function nntest.Bias()
      print('\n')
      local iW = 16
      local iH = 16
      local nPlanes = 3
      local batchSize = 8 
      local model = nn.Bias(nPlanes)
      model = model:cuda()
      local input = torch.CudaTensor(batchSize, nPlanes, iH, iW)
      local err,jf,jb = jac.testJacobian(model, input)
      print('error on state = ' .. err)
      local param,gradParam = model:parameters()
      local bias = param[1]
      local gradBias = gradParam[1]
      local err,jfp,jbp = jac.testJacobianParameters(model, input, bias, gradBias)
      print('error on bias = ' .. err)
      mytester:assertlt(err,precision, 'error on bias')
      print('\n')
   end
end

if test_complex_interp then
   function nntest.ComplexInterp()
      print('\n')
      local iW = 8
      local iH = 8
      local oW = 32
      local oH = 32
      local nInputs = 6
      local nSamples = 1
      global_debug1 = false
      global_debug2 = false
      local model = nn.ComplexInterp(iH, iW, oH, oW, 'bilinear')
      model = model:cuda()
      local input = torch.CudaTensor(nSamples,nInputs,iH,iW,2):normal()
      local err,jfc,jbc = jac.testJacobian(model, input)
      print('error on state =' .. err)
      mytester:assertlt(err,precision, 'error on state')
      print('\n')
   end
end


if test_interp then
   function nntest.Interp()
      print('\n')
      local k = 5
      local n = 32
      local nInputs = 6
      local nSamples = 2
      local model = nn.Interp(k,n,'bilinear')
      model = model:cuda()
      local input = torch.CudaTensor(nSamples,nInputs,k):normal()
      local err,jfc,jbc = jac.testJacobian(model, input)
      print('error on state =' .. err)
      mytester:assertlt(err,precision, 'error on state')
      print('\n')
   end
end









if test_interp_img then
   function nntest.InterpImage()
      print('\n')
      local iW = 8
      local iH = 8
      local oW = 32
      local oH = 32
      local nInputs = 6
      local nSamples = 2
      local model = nn.InterpImage(iH, iW, oH, oW, 'bilinear')
      model = model:cuda()
      local input = torch.CudaTensor(nSamples,nInputs,iH,iW,2):normal()
      local err,jfc,jbc = jac.testJacobian(model, input)
      print('error on state =' .. err)
      mytester:assertlt(err,precision, 'error on state')
      print('\n')
   end
end

if test_real then
   function nntest.Real()
      print('\n')
      local iW = 8
      local iH = 8
      local nInputPlanes = 3
      local batchSize = 2
      local model = nn.Real('real')
      model = model:cuda()
      local input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW,2)
      local err,jf,jb = jac.testJacobian(model, input)
      print('error on state = ' .. err)
      mytester:assertlt(err, precision, 'error on state')
      print('\n')
   end
end






if test_spectralconv then
   function nntest.SpectralConvolution()
      print('\n')
      torch.manualSeed(123)
      local interpType = 'bilinear'
      local dim = 200
      local subdim = 8
      local nInputPlanes = 8
      local nOutputPlanes = 26
      local batchSize = 3
      local model = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,dim, subdim, interpType)
      model = model:cuda()
      model:reset()
      local input = torch.CudaTensor(batchSize,nInputPlanes,dim):normal()
      err,jf,jb = jac.testJacobian(model, input)
      print('error on state =' .. err)
      mytester:assertlt(err,precision, 'error on state')
      
      local param,gradParam = model:parameters()
      local weight = param[1]
      local gradWeight = gradParam[1]
      err,jfp,jbp = jac.testJacobianParameters(model, input, weight, gradWeight)
      print('error on weight = ' .. err)
      mytester:assertlt(err,precision, 'error on weight')
      print('\n')
   end
end














if test_spectralconv_img then
   function nntest.SpectralConvolutionImage()
      print('\n')
      torch.manualSeed(123)
      local interpType = 'bilinear'
      local iW = 32
      local iH = 32
      local nInputPlanes = 2
      local nOutputPlanes = 2
      local batchSize = 2
      local sW = 4	
      local sH = 4
      local model = nn.SpectralConvolutionImage(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,interpType)
      model = model:cuda()
      model:reset()
      local input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):normal()
      local err,jf,jb = jac.testJacobian(model, input)
      print('error on state =' .. err)
      mytester:assertlt(err,precision, 'error on state')
      
      local param,gradParam = model:parameters()
      local weight = param[1]
      local gradWeight = gradParam[1]
      local err,jfp,jbp = jac.testJacobianParameters(model, input, weight, gradWeight)
      print('error on weight = ' .. err)
      mytester:assertlt(err,precision, 'error on weight')
      --[[
      param,gradParam = model:parameters()
      weight = param[1]
      gradWeight = gradParam[1]
      paramType='weight'
      local err = jac.testJacobianUpdateParameters(model, input, weight)
      print('error on weight [direct update] = ' .. err)
      --mytester:assertlt(err,precision, 'error on weight [direct update]')
      --]]
      print('\n')
   end
end

function run_timing()
   print('\n')
   print('******TIMING******')
   torch.manualSeed(123)
   local ntrials = 5
   local interpType = 'bilinear'
   local iW = 32
   local iH = 32
   local nInputPlanes = 96
   local nOutputPlanes = 256
   local batchSize = 128
   local sW = 8	
   local sH = 8
   local timer = torch.Timer()
   print('image dim = ' .. iH .. ' x ' .. iH)
   print('nInputPlanes = ' .. nInputPlanes)
   print('nOutputPlanes = ' .. nOutputPlanes)
   print('batchSize = ' .. batchSize)

   if test_spectralconv_img then
      model = nn.SpectralConvolutionImage(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,interpType)
      model = model:cuda()
      input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):zero()
      gradOutput = torch.CudaTensor(batchSize,nOutputPlanes,iH,iW)
      print('------SPECTRALCONVOLUTION------')
      for i = 1,ntrials do
         print('trial' .. i)
         timer:reset()
         model:forward(input)
         cutorch.synchronize()
         print('Time for forward : ' .. timer:time().real)

         timer:reset()
         model:updateGradInput(input,gradOutput)
         cutorch.synchronize()
         print('Time for updateGradInput : ' .. timer:time().real)

         timer:reset()
         model:accGradParameters(input,gradOutput)
         cutorch.synchronize()
         print('Time for accGradParameters : ' .. timer:time().real)
      end
   end

   if test_real then
      print('\n------REAL------\n')
      model2 = nn.Real('mod'):cuda()
      input2 = model.output:clone()
      gradOutput2 = torch.CudaTensor(batchSize, nOutputPlanes, iH, iW):zero()
      for i = 1,ntrials do
         timer:reset()
         model2:updateOutput(model.output)
         cutorch.synchronize()
         print('updateOutput : ' .. timer:time().real)

         timer:reset()
         model2:updateGradInput(input2,gradOutput2)
         cutorch.synchronize()
         print('updateGradInput : ' .. timer:time().real)
      end
   end

   if test_complex_interp then
      print('\n------COMPLEX_INTERP------')
      model3 = nn.ComplexInterp(sH,sW,iH,iW,'bilinear'):cuda()
      weights = torch.CudaTensor(nOutputPlanes, nInputPlanes, sH, sW, 2)
      for i = 1,ntrials do
         timer:reset()
         model3:updateOutput(weights)
         cutorch.synchronize()
         print('updateOutput : ' .. timer:time().real)
         gradWeights = model3.output:clone()
         timer:reset()
         model3:updateGradInput(weights,gradWeights)
         cutorch.synchronize()
         print('updateGradInput : ' .. timer:time().real)
      end
   end

   if test_interp then
      print('\n------INTERP------')
      model3 = nn.Interp(sH,sW,iH,iW,'bilinear'):cuda()
      weights = torch.CudaTensor(nOutputPlanes, nInputPlanes, sH, sW, 2)
      for i = 1,ntrials do
         timer:reset()
         model3:updateOutput(weights)
         cutorch.synchronize()
         print('updateOutput : ' .. timer:time().real)
         gradWeights = model3.output:clone()
         timer:reset()
         model3:updateGradInput(weights,gradWeights)
         cutorch.synchronize()
         print('updateGradInput : ' .. timer:time().real)
      end
   end

   if test_crop then
      print('\n--------CROP-------------')
      input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):zero()
      gradOutput = torch.CudaTensor(batchSize,nOutputPlanes,iH,iW)
      model4 = nn.Crop(iH,iW,2,2,false):cuda()
      for i = 1,ntrials do
         timer:reset()
         model4:updateOutput(input)
         cutorch.synchronize()
         print('updateOutput : ' .. timer:time().real)
         gradOutput = model4.output:clone()
         timer:reset()
         model4:updateGradInput(input, gradOutput)
         print('updateGradInput : ' .. timer:time().real)
      end
   end
end

mytester:add(nntest)
if test_correctness then
   mytester:run()
end

if test_time then 
   run_timing()
end



-- test frequency convolution module

require 'cunn'
require 'cucomplex'
require 'HermitianInterp'
require 'ComplexInterp'
require 'Modulus'
dofile('SpectralConvolution.lua')
dofile('Jacobian2.lua')
dofile('utils.lua')
cufft = dofile('cufft/cufft.lua')

torch.manualSeed(123)
cutorch.setDevice(2)
local test_correctness = false
local test_time = true
local mytester = torch.Tester()
local jac = nn.Jacobian
local sjac
local nntest = {}
local precision = 1e-1


function nntest.ComplexInterp()
	local iW = 4
	local iH = 4
	local oW = 16
	local oH = 16
	local nInputs = 4
	
	model = nn.ComplexInterp(iH, iW, oH, oW)
	model = model:cuda()
	input = torch.CudaTensor(nInputs,iH,iW,2):normal()
    --print('forward')
    --out = model:forward(input)
    --out = out:float():squeeze()
    --print('back')
    --gradOutput = torch.CudaTensor(nInputs, oH, oW, 2):normal()
    --g1=model:updateGradInput(input, gradOutput):clone()
    --model2 = model:float()
    --out2 = model2:forward(input:float()):squeeze()
    --g2=model:updateGradInput(input:float(), gradOutput:float()):clone()
	err,jf,jb = jac.testJacobian(model, input)
	print('error on state =' .. err)
	--mytester:assertlt(err,precision, 'error on state')
end

nntest.ComplexInterp()

function nntest.Modulus()
   local iW = 32
   local iH = 32
   local nInputPlanes = 16
   local batchSize = 16
   model = nn.Sequential()
   model:add(nn.Modulus())
   model = model:cuda()
   input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW,2)
   err,jf,jb = jac.testJacobian(model, input)
   print('error on state = ' .. err)
   --mytester:assertlt(err, precision, 'error on state')
   input = torch.CudaTensor(128,256,32,32,2)
   model = nn.Sequential()
   model:add(nn.Modulus())
   model = model:cuda()
   timer = torch.Timer():reset()
   out = model:forward(input)
   cutorch.synchronize()
   print('fprop: ' .. timer:time().real) timer:reset()
   gradOutput = out:clone()
   timer:reset()
   gradInput = model:updateGradInput(input, gradOutput)
   cutorch.synchronize()
   print('bprop: ' .. timer:time().real)
end

--nntest.Modulus()


function nntest.HermitianInterp()
	local sW = 4
	local sH = 4
	local iW = 8/2+1
	local iH = 8
	local nInputs = 1
	
	local model = nn.HermitianInterp(sH,sW,iH,iW)
	model = model:cuda()
	local input = torch.CudaTensor(1,nInputs,sH,sW,2):normal()
	
	local err = jac.testJacobian(model, input)
	print('error on state =' .. err)
	mytester:assertlt(err,precision, 'error on state')

end

function nntest.SpectralConvolution()
    torch.manualSeed(123)
    -- set to false if we assume weights are hermitian, true otherwise
    local full_weights = true
	local iW = 16
	local iH = 16
	local nInputPlanes = 1
	local nOutputPlanes = 1
	local batchSize = 1
	local sW = 8	
	local sH = 8
	model = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,full_weights)
	model = model:cuda()
	model:reset()
	input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):normal()
    model.bias:zero()

	err,jf,jb = jac.testJacobian(model, input)
	print('error on state =' .. err)
	mytester:assertlt(err,precision, 'error on state')
	
	local err = jac.testJacobianParameters(model, input, model.bias, model.gradBias)
	print('error on bias = ' .. err)
	mytester:assertlt(err,precision, 'error on bias')

	param,gradParam = model:parameters()
	weight = param[1]
	gradWeight = gradParam[1]
	err,jfp,jbp = jac.testJacobianParameters(model, input, weight, gradWeight)
	print('error on weight = ' .. err)
	mytester:assertlt(err,precision, 'error on weight')

	param,gradParam = model:parameters()
	weight = param[1]
	gradWeight = gradParam[1]
	paramType='weight'
	local err = jac.testJacobianUpdateParameters(model, input, weight)
	print('error on weight [direct update] = ' .. err)
	mytester:assertlt(err,precision, 'error on weight [direct update]')

    --[[
	bias = param[2]
    gradBias = gradParam[2]
	paramType = 'bias'
	err,jf,jb = jac.testJacobianUpdateParameters(model, input, bias)
	print('error on bias [direct update] = ' .. err)
	mytester:assertlt(err,precision, 'error on bias [direct update]')
    --]]

end


function run_timing()
	print('\n')
    print('******TIMING******')
    torch.manualSeed(123)
    local ntrials = 5
    local full_weights = true
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
	model = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,full_weights)
	model = model:cuda()
	--model:reset()
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

    print('\n------MODULUS------\n')
    model2 = nn.Modulus()
    input2 = model.output:clone()
    gradOutput2 = torch.CudaTensor(batchSize, nOutputPlanes, iH, iW):zero()
    for i = 1,ntrials do
       timer:reset()
       model:updateOutput(model.output)
       cutorch.synchronize()
       print('updateOutput : ' .. timer:time().real)

       timer:reset()
       model:updateGradInput(input2,gradOutput2)
       cutorch.synchronize()
       print('updateGradInput : ' .. timer:time().real)
    end

    print('\n------COMPLEX_INTERP------')
	model3 = nn.ComplexInterp(iH,iW,iH,iW,false,false):cuda()
    weights = torch.CudaTensor(nOutputPlanes, nInputPlanes, sH, sW, 2)
    for i = 1,ntrials do
       timer:reset()
       model3:updateOutput(weights)
       cutorch.synchronize()
       print('updateOutput : ' .. timer:time().real)
       gradWeights = model3.output:clone()
       timer:reset()
       model:updateGradInput(weights,gradWeights)
       cutorch.synchronize()
       print('updateGradInput : ' .. timer:time().real)
    end

end

mytester:add(nntest)
if test_correctness then
   mytester:run()
end

if test_time then 
   run_timing()
end



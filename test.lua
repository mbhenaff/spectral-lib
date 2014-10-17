-- test frequency convolution module

require 'cunn'
require 'cucomplex'
require 'HermitianInterp'
require 'ComplexInterp'
dofile('SpectralConvolution.lua')
dofile('Jacobian2.lua')
dofile('utils.lua')
cufft = dofile('cufft/cufft.lua')

torch.manualSeed(123)
cutorch.setDevice(3)
local test_correctness = true
local test_time = true
local mytester = torch.Tester()
local jac = nn.Jacobian
local sjac
local nntest = {}
local precision = 1e-1


function nntest.ComplexInterp()
	print('\n')
	local iW = 4
	local iH = 4
	local oW = 8
	local oH = 8
	local nInputs = 2
	
	local model = nn.ComplexInterp({iH,iW},{oH,oW},false,false)
	model = model:cuda()
	local input = torch.CudaTensor(nInputs,iH,iW,2):normal()
	
	local err = jac.testJacobian(model, input)
	print('error on state =' .. err)
	mytester:assertlt(err,precision, 'error on state')

end

function nntest.HermitianInterp()
	print('\n')
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
	print('\n')
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
    torch.manualSeed(123)
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
	model:reset()
	input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW)
    gradOutput = torch.CudaTensor(batchSize,nOutputPlanes,iH,iW)
    for i = 1,1 do
       print('***********************')
    timer:reset()
    model:updateOutput(input)
    print('Time for updateOutput : ' .. timer:time().real)

    timer:reset()
    model:updateGradInput(input,gradOutput)
    print('Time for updateGradInput : ' .. timer:time().real)

    timer:reset()
    model:accGradParameters(input,gradOutput)
    print('Time for accGradParameters : ' .. timer:time().real)
    end

end

mytester:add(nntest)
if test_correctness then
   mytester:run()
end

if test_time then 
   run_timing()
end



-- scripts for debugging

require 'cunn'
require 'cucomplex'
require 'ComplexInterp'
dofile('SpectralConvolution.lua')
dofile('Jacobian2.lua')
dofile('utils.lua')
cufft = dofile('cufft/cufft.lua')

torch.manualSeed(123)


local mytester = torch.Tester()
local jac = nn.Jacobian
local sjac
local nntest = {}
local precision = 1e-2


function make_hermitian_weights(nOutputPlanes,nInputPlanes,iH,iW)
    local spatial_weights = torch.zeros(nOutputPlanes,nInputPlanes,iH,iW,2)
    spatial_weights:select(5,1):copy(torch.randn(nOutputPlanes,nInputPlanes,iH,iW))
    local weights = torch.CudaTensor(nOutputPlanes,nInputPlanes,iH,iW,2):zero()
    cufft.fft2d_c2c(spatial_weights:cuda(),weights,1)
    return weights
end











--[[
function nntest.ComplexInterp()
	print('\n')
	local iW = 4
	local iH = 1
	local oW = 8
	local oH = 1
	local nInputs = 1
	
	local model = nn.ComplexInterp({iH,iW},{oH,oW},true,true)
	model = model:cuda()
	local input = torch.CudaTensor(nInputs,iH,iW,2):normal()
	
	local err = jac.testJacobian(model, input)
	print('error on state =' .. err)
	mytester:assertlt(err,precision, 'error on state')

end
--]]

--[[
function nntest.SpectralvSpatial()
   iW = 16
   iH = 8
   sW = 4
   sH = 4
   kW = 3
   kH = 3
   batchSize = 1
   nInputPlanes = 1
   nOutputPlanes = 1

   model=nn.SpectralConvolution(batchSize, nInputPlanes, nOutputPlanes, iH, iW, sH, sW):cuda()
   model.bias:zero()
   print(model)
   data = torch.randn(batchSize,nInputPlanes,iH,iW):cuda()

   -- convolution kernels
   kernel = torch.randn(nOutputPlanes,nInputPlanes,kH,kW)
   kernelPadded = torch.zeros(nOutputPlanes,nInputPlanes,iH,iW):cuda()
   kernelPadded[{{},{},{1,kH},{1,kW}}]:copy(kernel)
   -- put in spectral domain
   kernelSpec=torch.CudaTensor(nOutputPlanes,nInputPlanes,iH,iW/2+1,2)
   cufft.fft2d(kernelPadded,kernelSpec)
   model.weight:copy(kernelSpec)

   -- make regular convolution module (remember to reverse kernel)
   model2 = nn.SpatialConvolution(nOutputPlanes,nInputPlanes, kW,kH)
   model2.bias:zero()
   model2.weight:copy(kernel)


   out1 = model:forward(data)
   out2 = model2:forward(data:double())
   --out1 = out1:double():squeeze()
   --out1 = out1[{{},{},{3,iW}}]
   --out2 = out2:squeeze()

   --out1 = out1[{{},{3,iW}}]

   gradInput1 = model:updateGradInput(data,out1)
   gradInput2 = model2:updateGradInput(data:double(),out2)

   --gradWeight = model:accGradParameters(data, out,1)
end

nntest.SpectralvSpatial()
--]]



function nntest.SpectralConvolution()
	print('\n')
    torch.manualSeed(123)
    local full_weights = true
	local iW = 16
	local iH = 16
 	local nInputPlanes = 1
	local nOutputPlanes = 1
	local batchSize = 1
	local sW = 4	
	local sH = 4
	model = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,full_weights)
	model = model:cuda()
	model:reset()
	input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):normal()
    model.bias:zero()
    if global_debug then
       weights = make_hermitian_weights(nOutputPlanes,nInputPlanes,iH,iW)
       model.weight:copy(weights)
    end

	local err = jac.testJacobian(model, input)
	print('error on state =' .. err)
	mytester:assertlt(err,precision, 'error on state')
	
	local err = jac.testJacobianParameters(model, input, model.bias, model.gradBias)
	print('error on bias = ' .. err)
	mytester:assertlt(err,precision, 'error on bias')

	param,gradParam = model:parameters()
	weight = param[1]
	gradWeight = gradParam[1]
	local err = jac.testJacobianParameters(model, input, weight, gradWeight)
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
	paramType = 'bias'
	local err = jac.testJacobianUpdateParameters(model, input, model.bias)
	print('error on bias [direct update] = ' .. err)
	mytester:assertlt(err,precision, 'error on bias [direct update]')
	--]]	
end



function nntest.SpectralConvolution2()
	print('\n')
    torch.manualSeed(123)
	local iW = 8
	local iH = 8
	local nInputPlanes = 1
	local nOutputPlanes = 1
	local batchSize = 1
	local sW = 4	
	local sH = 4
    weights = make_hermitian_weights(nOutputPlanes,nInputPlanes,iH,iW)
  
    global_debug = true
	model1 = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW)
	model1 = model1:cuda()
	model1:reset()
	input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):normal()
    model1.bias:zero()
    model1.weight:copy(weights)

    global_debug = false
	model2 = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW)
	model2 = model2:cuda()
	model2:reset()
    model2.bias:zero()
    model2.weight:copy(model1.weight[{{},{},{},{1,iW/2+1},{}}])
    print(input:norm())
    global_debug = true
    out1 = model1:updateOutput(input:clone()):clone()
    g1 = model1:updateGradInput(input,out1:clone())
    global_debug = false
    print(input:norm())
    out2 = model2:updateOutput(input:clone()):clone()
    g2 = model2:updateGradInput(input,out1:clone())
 end

function nntest.SpectralConvolution3()
	print('\n')
    torch.manualSeed(123)
    global_debug = true
    global_print = false
	local iW =8
	local iH = 8
	local nInputPlanes = 1
	local nOutputPlanes = 1
	local batchSize = 1
	local sW = 4	
	local sH = 4
    weights = make_hermitian_weights(nOutputPlanes,nInputPlanes,iH,iW)

    
	model1 = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,false)
	model1 = model1:cuda()
	model1:reset()
    model1.bias:zero()
    model1.weight:copy(weights[{{},{},{},{1,iW/2+1},{}}])

    -- record full set of weights
	model2 = nn.SpectralConvolution(batchSize,nInputPlanes,nOutputPlanes,iH,iW,sH,sW,true)
	model2 = model2:cuda()
	model2:reset()
    model2.bias:zero()
    model2.weight:copy(weights)
   

	input = torch.CudaTensor(batchSize,nInputPlanes,iH,iW):normal()

    model1:forward(input)
    model2:forward(input)
    -- check they produce the same output
    out1 = model1:updateOutput(input):float()
    out2 = model2:updateOutput(input):float()
    err = torch.max(torch.abs(out1:float()-out2:float()))
    print('error on output = ' .. err)

    -- compare forward jacobian
    --jac1,jac2 = nn.Jacobian.forward2(model1,model2,input)
    --jacb1,jacb2 = nn.Jacobian.backward2(model1,model2,input)
    --jac1 = nn.Jacobian.forward(model1,input)
    --jac2 = nn.Jacobian.forward(model2,input)
    if true then

	err,jf1,jb1 = jac.testJacobian(model1, input)
	print('error on state for model 1 =' .. err)
	err,jf2,jb2 = jac.testJacobian(model2, input)
	print('error on state for model 2 =' .. err)
	--mytester:assertlt(err,precision, 'error on state')

    -- check they produce the same output
    out1 = model1:updateOutput(input):float()
    out2 = model2:updateOutput(input):float()
    err = torch.max(torch.abs(out1:float()-out2:float()))
    print('error on output = ' .. err)

    end

    if false then
	model = model2
	err,jf2,jb2 = jac.testJacobian(model, input)
	print('error on state =' .. err)
	err = jac.testJacobianParameters(model, input, model.bias, model.gradBias)
	print('error on bias = ' .. err)
	--mytester:assertlt(err,precision, 'error on bias')

	param,gradParam = model:parameters()
	weight = param[1]
	gradWeight = gradParam[1]
	err,jfp,jbp = jac.testJacobianParameters(model, input, weight, gradWeight)
	print('error on weight = ' .. err)
	--mytester:assertlt(err,precision, 'error on weight')

	param,gradParam = model:parameters()
	weight = param[1]
	gradWeight = gradParam[1]
	paramType='weight'
	local err = jac.testJacobianUpdateParameters(model, input, weight)
	print('error on weight [direct update] = ' .. err)

    end
	--mytester:assertlt(err,precision, 'error on weight [direct update]')
 end



nntest.SpectralConvolution3()













--nntest.SpectralConvolution3()
mytester:add(nntest)
--mytester:run()



--[[
--]]


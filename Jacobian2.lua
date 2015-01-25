nn.Jacobian = {}

-- Modified Jacobian function to deal with the spectral conv module. 
-- Note, it is important to remember that the forward method pushes the weights 
-- through the interpolation module. Therefore we use updateOutput for unit 
-- tests on gradients, and forward for the tests on weights. 

function nn.Jacobian.backward (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   if doparam == 1 then
      module:forward(input)
   else
      module:updateOutput(input)
   end
   local dout = module.output.new():resizeAs(module.output)
   dout = dout:cuda()
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()

   for i=1,sdout:nElement() do
      dout:zero()
      sdout[i] = 1
	  if doparam == 1 then
		  module:backward(input,dout)
		  jacobian:select(2,i):copy(dparam)
       else
		  module:zeroGradParameters()
		  local din = module:updateGradInput(input, dout)
		  module:accGradParameters(input, dout)
		  jacobian:select(2,i):copy(din)
	  end
   end
   return jacobian
end

function nn.Jacobian.backwardUpdate (module, input, param)
   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()
   -- original param
   local params = module:parameters()
   local origparams = {}
   for j=1,#params do
      table.insert(origparams, params[j]:clone())
   end

   for i=1,sdout:nElement() do
      for j=1,#params do
         params[j]:copy(origparams[j])
      end
      dout:zero()
      sdout[i] = 1
	  if paramType == 'bias' then
		local din = module:updateGradInput(input, dout)
		module:accUpdateGradParameters(input, dout, 1)
		jacobian:select(2,i):copy(param)
	  elseif paramType == 'weight' then
		module:backward(input,dout,1)
		jacobian:select(2,i):copy(param)
		jacobian:select(2,i):add(-1,module.gradWeightPreimage:double())
	  end
   end
   for j=1,#params do
      params[j]:copy(origparams[j])
   end

   return jacobian
end

function nn.Jacobian.forward(module, input, param)
   local doparam = 0
   if param then 
      doparam = 1 
   end
   param = param or input
   -- perturbation amount
   local small = 1e-4
   -- 1D view of input
   local sin = param.new(param):resize(param:nElement())
   -- jacobian matrix to calculate
   local jacobian
   if doparam == 1 then 
      jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   else
      jacobian = torch.Tensor():resize(param:nElement(),module:updateOutput(input):nElement())
   end

   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      if doparam == 1 then
         outa:copy(module:forward(input))
      else
         outa:copy(module:updateOutput(input))
      end
      sin[i] = sin[i] + 2*small
      if doparam == 1 then
         outb:copy(module:forward(input))
      else
         outb:copy(module:updateOutput(input))
      end
      sin[i] = sin[i] - small
      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
   end
   return jacobian
end

-- debug
function nn.Jacobian.forward2(module1,module2, input, param)
   param = param or input
   -- perturbation amount
   local small = 1e-4
   -- 1D view of input
   local sin = param.new(param):resize(param:nElement())
   -- jacobian matrix to calculate
   local jacobian1 = torch.Tensor():resize(param:nElement(),module1:updateOutput(input):nElement())
   local jacobian2 = jacobian1:clone()

   local outa1 = torch.Tensor(jacobian1:size(2))
   local outb1 = torch.Tensor(jacobian1:size(2))
   local outa2 = torch.Tensor(jacobian2:size(2))
   local outb2 = torch.Tensor(jacobian2:size(2))

   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa1:copy(module1:updateOutput(input))
      outa2:copy(module2:updateOutput(input))
      print('err(outa1 - outa2) = ' .. torch.max(torch.abs(outa1-outa2)))
      --outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb1:copy(module1:updateOutput(input))
      outb2:copy(module2:updateOutput(input))
      print('err(outb1 - outb2) = ' .. torch.max(torch.abs(outb1-outb2)))
      --outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      diff1 = outb1 - outa1
      diff2 = outb2 - outa2
      print('err(diff1 - diff2) = ' .. torch.max(torch.abs(diff1 - diff2)))
      outb1:add(-1,outa1):div(2*small)
      outb2:add(-1,outa2):div(2*small)
      jacobian1:select(1,i):copy(outb1)
      jacobian2:select(1,i):copy(outb2)
   end
   print('err(jac1 - jac2) = ' .. torch.max(torch.abs(jacobian1 - jacobian2)))
   return jacobian1,jacobian2
end


function nn.Jacobian.backward2(module1,module2, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   module1:updateOutput(input)
   local dout1 = module1.output.new():resizeAs(module1.output)
   dout1 = dout1:cuda()
   -- 1D view
   local sdout1 = module1.output.new(dout1:storage(),1,dout1:nElement())
   -- jacobian matrix to calculate
   local jacobian1 = torch.Tensor(param:nElement(),dout1:nElement()):zero()

   module2:updateOutput(input)
   local dout2 = module2.output.new():resizeAs(module2.output)
   dout2 = dout2:cuda()
   -- 1D view
   local sdout2 = module2.output.new(dout2:storage(),1,dout2:nElement())
   -- jacobian matrix to calculate
   local jacobian2 = torch.Tensor(param:nElement(),dout2:nElement()):zero()

   for i=1,sdout1:nElement() do
      dout1:zero()
      sdout1[i] = 1
      dout2:zero()
      sdout2[i] = 1
	  if doparam == 1 then
		  module:backward(input,dout)
		  jacobian:select(2,i):copy(dparam)
	  else
		  module1:zeroGradParameters()
		  local din1 = module1:updateGradInput(input, dout1)
		  module1:accGradParameters(input, dout1)
		  jacobian1:select(2,i):copy(din1)

		  module2:zeroGradParameters()
		  local din2 = module2:updateGradInput(input, dout2)
		  module2:accGradParameters(input, dout2)
		  jacobian2:select(2,i):copy(din2)
          print('err(din1-din2) = ' .. torch.max(torch.abs(din1:float()-din2:float())))
	  end
   end
   return jacobian1,jacobian2
end



function nn.Jacobian.forwardUpdate(module, input, param)
   -- perturbation amount
   local small = 1e-4
   -- 1D view of input
   local sin =  param.new(param):resize(param:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
      jacobian:select(1,i):mul(-1)
      jacobian:select(1,i):add(sin[i])
   end
   return jacobian
end

function nn.Jacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   local jac_fprop = nn.Jacobian.forward(module,input)
   local jac_bprop = nn.Jacobian.backward(module,input)
   local error = jac_fprop-jac_bprop
   return error:abs():max(),jac_fprop,jac_bprop
end

function nn.Jacobian.testJacobianParameters (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))	
   local jac_fprop = nn.Jacobian.forward(module, input, param)
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local error = jac_fprop - jac_bprop
   return error:abs():max(),jac_fprop,jac_bprop
end

function nn.Jacobian.testJacobianUpdateParameters (module, input, param, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local params_bprop = nn.Jacobian.backwardUpdate(module, input, param)
   local params_fprop = nn.Jacobian.forwardUpdate(module, input, param)
   local error = params_fprop - params_bprop
   return error:abs():max(),params_fprop,params_bprop
end

function nn.Jacobian.testIO(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = module.output:clone():copy(torch.rand(module.output:nElement()):mul(inrange):add(minval))
   module:zeroGradParameters()
   module:updateGradInput(input,go)
   module:accGradParameters(input,go)

   local fo = module.output:clone()
   local bo = module.gradInput:clone()

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:zeroGradParameters()
   m:updateGradInput(input,go)
   m:accGradParameters(input,go)
   -- cleanup
   os.remove('tmp.bin')

   local fo2 = m.output:clone()
   local bo2 = m.gradInput:clone()

   local errf = fo - fo2
   local errb = bo - bo2
   return errf:abs():max(), errb:abs():max()
end

function nn.Jacobian.testAllUpdate(module, input, weight, gradWeight)
   local gradOutput
   local lr = torch.uniform(0.1, 1)
   local errors = {}

   -- accGradParameters
   local maccgp = module:clone()
   local weightc = maccgp[weight]:clone()
   maccgp:forward(input)
   gradOutput = torch.rand(maccgp.output:size())
   maccgp:zeroGradParameters()
   maccgp:updateGradInput(input, gradOutput)
   maccgp:accGradParameters(input, gradOutput)
   maccgp:updateParameters(lr)
   errors["accGradParameters"] = (weightc-maccgp[gradWeight]*lr-maccgp[weight]):norm()
   
   -- accUpdateGradParameters
   local maccugp = module:clone()
   maccugp:forward(input)
   maccugp:updateGradInput(input, gradOutput)
   maccugp:accUpdateGradParameters(input, gradOutput, lr)
   errors["accUpdateGradParameters"] = (maccugp[weight]-maccgp[weight]):norm()

   -- shared, accGradParameters
   local macsh1 = module:clone()
   local macsh2 = module:clone()
   macsh2:share(macsh1, weight)
   macsh1:forward(input)
   macsh2:forward(input)
   macsh1:zeroGradParameters()
   macsh2:zeroGradParameters()
   macsh1:updateGradInput(input, gradOutput)
   macsh2:updateGradInput(input, gradOutput)
   macsh1:accGradParameters(input, gradOutput)
   macsh2:accGradParameters(input, gradOutput)
   macsh1:updateParameters(lr)
   macsh2:updateParameters(lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macsh1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macsh2[weight]):norm()
   errors["accGradParameters [shared]"] = err
   
   -- shared, accUpdateGradParameters
   local macshu1 = module:clone()
   local macshu2 = module:clone()
   macshu2:share(macshu1, weight)
   macshu1:forward(input)
   macshu2:forward(input)
   macshu1:updateGradInput(input, gradOutput)
   macshu2:updateGradInput(input, gradOutput)
   macshu1:accUpdateGradParameters(input, gradOutput, lr)
   macshu2:accUpdateGradParameters(input, gradOutput, lr)
   local err = (weightc-maccgp[gradWeight]*(lr*2)-macshu1[weight]):norm()
   err = err + (weightc-maccgp[gradWeight]*(lr*2)-macshu2[weight]):norm()
   errors["accUpdateGradParameters [shared]"] = err

   return errors
end

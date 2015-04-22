-- General interpolation module. Images and weights are resized to be n^2 and k^2 respectively. 
-- The interpolation map is a n^2 x k^2 matrix. 
require 'nn'

local LearnableInterp2D, parent = torch.class('nn.LearnableInterp2D', 'nn.Module')

function LearnableInterp2D:__init(iH, iW, oH, oW, interpType)
   parent.__init(self)
   assert(iH == iW and oH == oW)
   self.iH = iH
   self.iW = iW
   self.oH = oH
   self.oW = oW
   self.interpType = interpType
   self.cntr = 0
   if self.interpType == 'spatial' then
      self.weight = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/spatial_kernel_' .. iH .. '_' .. oH .. '.th'):float()
   else
      self.weight = compute_interpolation_matrix(iH, oH, interpType)
   end
   self.gradWeight = torch.Tensor(self.weight:size()):zero()
   self.mask = torch.Tensor(self.weight:size()):fill(1)
   for i = 1,self.weight:size(1) do 
      for j = 1,self.weight:size(2) do 
         if math.abs(self.weight[i][j]) < 1e-5 then
            self.mask[i][j] = 0
         end
      end
   end
   self.originalKernel = self.weight:clone()
end

--[[
function LearnableInterp2D:reset()
   if self.interpType == 'spatial' then
      self.weight = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/spatial_kernel_' .. iH .. '_' .. oH .. '.th'):float()
   else
      self.weight = compute_interpolation_matrix(iH, oH, interpType)
   end
   self.gradWeight = torch.Tensor(self.weight:size()):zero()
   self.mask = torch.Tensor(self.weight:size()):fill(1)
   for i = 1,self.weight:size(1) do 
      for j = 1,self.weight:size(2) do 
         if math.abs(self.weight[i][j]) < 1e-5 then
            self.mask[i][j] = 0
         end
      end
   end
end
--]]



function LearnableInterp2D:updateOutput(input)

   self.cntr = self.cntr + 1
   -- rescale the weights to the right nuclear norm
   if self.cntr % 100 == 0 then
      local s = estimate_scaling(self.originalKernel, self.weight)
      print('rescaling, s = ' .. s)
      print(self.weight:norm())
      self.weight:mul(s)
      collectgarbage()
   end
   self.weight:cmul(self.mask)
   local d1 = input:size(1)
   local d2 = input:size(2)
   input:resize(d1*d2, self.iH*self.iW*2)
   self.output:resize(d1*d2, self.oH*self.oW*2)
   self.output:zero()
   self.output:addmm(input,self.weight)
   input:resize(d1,d2,self.iH,self.iW,2)
   self.output:resize(d1,d2,self.oH,self.oW,2)
   return self.output
end
   
function LearnableInterp2D:updateGradInput(input, gradOutput)
   self.weight:cmul(self.mask)
   local d1 = input:size(1)
   local d2 = input:size(2)
   gradOutput:resize(d1*d2, self.oH*self.oW*2)
   self.gradInput:resize(d1*d2, self.iH*self.iW*2)
   self.gradInput:zero()
   self.gradInput:addmm(gradOutput, self.weight:t())
   self.gradInput:resize(d1,d2,self.iH,self.iW,2)
   gradOutput:resize(d1,d2,self.oH,self.oW,2)
   return self.gradInput
end

function LearnableInterp2D:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   local d1 = input:size(1)
   local d2 = input:size(2)
   gradOutput:resize(d1*d2, self.oH*self.oW*2)
   input:resize(d1*d2,self.iH*self.iW*2)
   self.gradWeight:zero()
   self.gradWeight:addmm(scale, input:t(), gradOutput)
   self.gradWeight:cmul(self.mask)
   input:resize(d1, d2, self.iH, self.iW, 2)
   gradOutput:resize(d1, d2, self.oH, self.oW, 2)
end

function compute_interpolation_matrix(k,n,interpType)
   local K = torch.FloatTensor(2*k^2, 2*n^2)
   local model = nn.ComplexInterp(k,k,n,n,interpType):float()
   local input = torch.FloatTensor(1,k,k,2)
   local cntr = 1
   for i = 1,k do
      for j = 1,k do 
         for l = 1,2 do 
            input:zero()
            input[1][i][j][l] = 1
            out = model:forward(input)
            K[{cntr,{}}]:copy(out:resize(2*n^2))
            cntr = cntr + 1
         end
      end
   end
   -- scale so it has similar norm to FFT matrix
   local FFTmat = interpKernel(k,n,'spatial2D')
   local scale = estimate_scaling(FFTmat, K)
   print('scaling factor: ' .. scale)
   K:mul(scale)
   return K,scale
end

-- estimate a scaling factor for matrix M2 so that it has a similar matrix norm as M1
function estimate_scaling(M1, M2, npts)
   local npts = npts or 1000
   local k = M1:size(1) 
   local n = M1:size(2)
   local out1
   local out2
   local input
   if M2:type() == 'torch.CudaTensor' then
      input = torch.rand(npts,k):cuda()
      out1 = torch.CudaTensor(npts,n):zero()
      out2 = torch.CudaTensor(npts,n):zero()
   else
      input = torch.rand(npts,k):float()
      out1 = torch.FloatTensor(npts,n):zero()
      out2 = torch.FloatTensor(npts,n):zero()
   end

   for i = 1,npts do 
      input[i]:mul(1/input[i]:norm())
   end
   out1:addmm(input,M1)
   out2:addmm(input,M2)
   
   local d1 = out1:norm(2,2)
   local d2 = out2:norm(2,2)
   return torch.max(d1) / torch.max(d2)
end





   
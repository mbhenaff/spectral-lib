-- General interpolation module. Images and weights are resized to be n^2 and k^2 respectively. 
-- The interpolation map is a n^2 x k^2 matrix. 
require 'nn'

local Interp3D, parent = torch.class('nn.Interp3D', 'nn.Module')

function Interp3D:__init(iF, iH, iW, oF, oH, oW, interpType)
   parent.__init(self)
   self.iF = iF
   self.iH = iH
   self.iW = iW
   self.oF = oF
   self.oH = oH
   self.oW = oW
   if interpType == 'spatial' then
      self.kernel = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/spatial_kernel_' .. iH .. '_' .. oH .. '.th'):float()
   else
      require 'ComplexInterp'
      self.kernel = compute_interpolation_matrix(iH, oH, interpType)
   end
   self.kernelT = self.kernel:t():clone()
   self.kernelFeat = interpKernel(iF,oF,interpType)
   self.kernelFeatT = self.kernelFeat:t():clone()
   self.bufferIn = torch.FloatTensor()
   self.bufferOut = torch.FloatTensor()
end

function Interp3D:featureTransform(input, M)
   local nSamples = input:size(1)
   local nInputPlanes = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   local nOutputPlanes = M:size(1)
   self.bufferIn:resize(input:size())
   self.bufferIn:copy(input)
   self.bufferIn = self.bufferIn:transpose(1,2):contiguous():resize(nInputPlanes,nSamples*iH*iW*2)
   self.bufferOut:resize(nOutputPlanes, nSamples*iH*iW*2)
   self.bufferOut:zero()   
   self.bufferOut:addmm(M, self.bufferIn)
   self.bufferOut:resize(nOutputPlanes,nSamples,iH,iW,2)
   self.bufferOut = self.bufferOut:transpose(1,2):contiguous()
   return self.bufferOut
end

function Interp3D:updateOutput(input)
  -- print(input:size())
   local d1 = input:size(1)
   local d2 = input:size(2)
   input:resize(d1*d2, self.iH*self.iW*2)
   self.output:resize(d1*d2, self.oH*self.oW*2)
   self.output:zero()
   self.output:addmm(input,self.kernel)
   input:resize(d1,self.iF,self.iH,self.iW,2)
   self.output:resize(d1,self.iF,self.oH,self.oW,2)
   self.output = self:featureTransform(self.output, self.kernelFeatT):clone()
--   print(self.output:size())
   return self.output
end
   
function Interp3D:updateGradInput(input, gradOutput)
   --print('gradOutput')
   --print(gradOutput:size())
   local d1 = gradOutput:size(1)
   local d2 = gradOutput:size(2)
   gradOutput:resize(d1*d2, self.oH*self.oW*2)
   self.gradInput:resize(d1*d2, self.iH*self.iW*2)
   self.gradInput:zero()
   self.gradInput:addmm(gradOutput, self.kernelT)
   self.gradInput:resize(d1,d2,self.iH,self.iW,2)
   gradOutput:resize(d1,d2,self.oH,self.oW,2)
   self.gradInput = self:featureTransform(self.gradInput, self.kernelFeat):clone()
   --print('gradInput')
   --print(self.gradInput:size())
   return self.gradInput
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
   return K
end

-- estimate a scaling factor for matrix M2 so that it has a similar matrix norm as M1
function estimate_scaling(M1, M2)
   local k = M1:size(1) 
   local n = M1:size(2)
   local s = 1000
   local input = torch.rand(s,k):float()
   for i = 1,s do 
      input[i]:mul(1/input[i]:norm())
   end
   local out1 = input*M1
   local out2 = input*M2
   local d1 = out1:norm(2,2)
   local d2 = out2:norm(2,2)
   return torch.max(d1) / torch.max(d2)
end





   
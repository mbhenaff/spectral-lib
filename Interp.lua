

local Interp, parent = torch.class('nn.Interp', 'nn.Module')

function Interp:__init(k, n, interpType)
   parent.__init(self)
   self.k = k
   self.n = n
   self.kernel = interpKernel(k,n,'bilinear')
  -- self.kernel = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/spatial_kernel_' .. k .. '_' .. n .. '.th'):float()
   self.kernelT = self.kernel:t():clone()
end

function Interp:updateOutput(input)
   local d1 = input:size(1)
   local d2 = input:size(2)
   input:resize(d1*d2, self.k)
   self.output:resize(d1*d2, self.n)
   self.output:zero()
   self.output:addmm(input,self.kernel)
   input:resize(d1,d2,self.k)
   self.output:resize(d1,d2,self.n)
   return self.output
end
   
function Interp:updateGradInput(input, gradOutput)
   local d1 = input:size(1)
   local d2 = input:size(2)
   gradOutput:resize(d1*d2, self.n)
   self.gradInput:resize(d1*d2, self.k)
   self.gradInput:zero()
   self.gradInput:addmm(gradOutput, self.kernelT)
   self.gradInput:resize(d1,d2,self.k)
   gradOutput:resize(d1,d2,self.n)
   return self.gradInput
end
   
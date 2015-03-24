
local Interp, parent = torch.class('nn.Interp', 'nn.Module')

function Interp:__init(k, n, interpType)
   parent.__init(self)
   self.k = k
   self.n = n
   self.kernel = interpKernel(k,n,'bilinear')
   local norm = estimate_norm(self.kernel)
   -- scale this so gradients are comparable with spatial kernel
   local scale = math.sqrt(2)*math.sqrt(n) / norm
   print('scaling factor: ' .. scale)
   self.kernel:mul(scale)
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
   

function estimate_norm(M1)
   local k = M1:size(1) 
   local n = M1:size(2)
   local s = 1000
   local input = torch.rand(s,k):float()
   for i = 1,s do 
      input[i]:mul(1/input[i]:norm())
   end
   local out1 = input*M1
   local d1 = out1:norm(2,2)
   return torch.max(d1)
end

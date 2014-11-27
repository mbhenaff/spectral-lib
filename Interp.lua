-- General interpolation module. Images and weights are resized to be n^2 and k^2 respectively. 
-- The interpolation map is a n^2 x k^2 matrix. 



local Interp, parent = torch.class('nn.Interp', 'nn.Module')

function Interp:__init(iH, iW, oH, oW, interpType)
   parent.__init(self)
   self.iH = iH
   self.iW = iW
   self.oH = oH
   self.oW = oW
   self.kernel = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/interp_kernels/spatial_kernel_' .. iH .. '_' .. oH .. '.th'):float()
   self.kernelT = self.kernel:t():clone()
end

function Interp:updateOutput(input)
   local d1 = input:size(1)
   local d2 = input:size(2)
   input:resize(d1*d2, self.iH*self.iW*2)
   self.output:resize(d1*d2, self.oH*self.oW*2)
   self.output:zero()
   self.output:addmm(input,self.kernel)
   input:resize(d1,d2,self.iH,self.iW,2)
   self.output:resize(d1,d2,self.oH,self.oW,2)
   return self.output
end
   
function Interp:updateGradInput(input, gradOutput)
   local d1 = input:size(1)
   local d2 = input:size(2)
   gradOutput:resize(d1*d2, self.oH*self.oW*2)
   self.gradInput:resize(d1*d2, self.iH*self.iW*2)
   self.gradInput:zero()
   self.gradInput:addmm(gradOutput, self.kernelT)
   self.gradInput:resize(d1,d2,self.iH,self.iW,2)
   gradOutput:resize(d1,d2,self.oH,self.oW,2)
   return self.gradInput
end
   
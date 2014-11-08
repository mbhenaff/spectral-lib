require 'nn'

local Modulus, parent = torch.class('nn.Modulus','nn.Module')

function Modulus:__init(dim)
   parent.__init(self)
   self.dim = dim
   self.outputtmp = torch.Tensor()
end

-- we assume inputs are of the form [batch x planes x iH x iW x 2]
function Modulus:updateOutput(input)
   local b = input:size(1)
   local p = input:size(2)
   local x = input:size(3)
   local y = input:size(4)
   input:resize(b*p,x,y,2)
   self.output:resize(b*p,x,y)
   self.output:copy(input:norm(2,4))
   input:resize(b,p,x,y,2)
   self.output:resize(b,p,x,y)
   return self.output
end

function Modulus:updateGradInput(input, gradOutput)
   if false then
   local b = input:size(1)
   local p = input:size(2)
   local x = input:size(3)
   local y = input:size(4)
   self.gradInput:resizeAs(input)
   self.gradInput:copy(input)
   gradOutput:resize(b,p,x,y,1)
   self.output:resize(b,p,x,y,1)
   self.gradInput:cmul(gradOutput:expandAs(input))
   --self.output2:resizeAs(self.output)
   --self.output2:copy(self.output)
   --self.gradInput:cdiv(self.output:expandAs(input))
   --self.gradInput:mul(2)
   gradOutput:resize(b,p,x,y)
   self.output:resize(b,p,x,y)
else
   cucomplex.modulus_updateGradInput(input, self.output, self.gradInput, gradOutput);
end
   return self.gradInput
end

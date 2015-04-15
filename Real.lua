-- make a complex input real by either taking the modulus, or discarding the imaginary part
require 'nn'

local Real, parent = torch.class('nn.Real','nn.Module')

function Real:__init(type)
   parent.__init(self)
   if type == 'mod' then 
      self.modulus = true
   elseif type == 'real' then
      self.modulus = false 
   else
      error('type must be mod or real')
   end
end

-- we assume inputs are of the form [batch x planes x iH x iW x 2]
function Real:updateOutput(input)
   local b = input:size(1)
   local p = input:size(2)
   local x = input:size(3)
   local y = input:size(4)
   input:resize(b*p,x,y,2)
   self.output:resize(b*p,x,y)
   if self.modulus then
      self.output:copy(input:norm(2,4))
   else
      self.output:copy(input:select(4,1))
   end
   input:resize(b,p,x,y,2)
   self.output:resize(b,p,x,y)
   return self.output
end

function Real:updateGradInput(input, gradOutput)
   if self.modulus then
      libspectralnet.modulus_updateGradInput(input, self.output, self.gradInput, gradOutput)
   else
      local b = input:size(1)
      local p = input:size(2)
      local x = input:size(3)
      local y = input:size(4)
      self.gradInput:resizeAs(input)
      self.gradInput:select(5,1):copy(gradOutput)
      self.gradInput:select(5,2):zero()
   end
   return self.gradInput
end

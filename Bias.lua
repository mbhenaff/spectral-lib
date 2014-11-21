require 'nn'

local Bias, parent = torch.class('nn.Bias', 'nn.Module')

function Bias:__init(nPlanes)
   parent.__init(self)
   self.bias = torch.Tensor(nPlanes)
   self.gradBias = torch.Tensor(nPlanes)
   self:reset()
end

function Bias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.bias:size(1))
   end
   self.bias:uniform(-stdv, stdv)
end

function Bias:updateOutput(input)
   self.output:resize(input:size())
   self.output:copy(input)
   cucomplex.bias_updateOutput(self.bias, self.output)
   return self.output
end


function Bias:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size())
   self.gradInput:copy(gradOutput)
   return self.gradInput
end

function Bias:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   cucomplex.bias_accGradParameters(self.gradBias, gradOutput, scale)
end


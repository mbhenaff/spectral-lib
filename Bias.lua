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
   local d1 = input:size(1)
   local d2 = input:size(2)
   local d3 = input:size(3)
   local resize = false 
   if input:nDimension() == 3 then
      resize = true
      input:resize(d1,d2,d3,1)
   end
   self.output:resize(input:size())
   self.output:copy(input)
   spectralcuda.bias_updateOutput(self.bias, self.output)
   if resize then 
      input:resize(d1,d2,d3)
      self.output:resize(input:size())
   end
   return self.output
end


function Bias:updateGradInput(input, gradOutput) 
   self.gradInput:resize(input:size())
   self.gradInput:copy(gradOutput)
   return self.gradInput
end

function Bias:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   local d1 = gradOutput:size(1)
   local d2 = gradOutput:size(2)
   local d3 = gradOutput:size(3)
   local resize = false 
   if gradOutput:nDimension() == 3 then
      resize = true
      gradOutput:resize(d1,d2,d3,1)
   end   
   spectralcuda.bias_accGradParameters(self.gradBias, gradOutput, scale)
   if resize then 
      gradOutput:resize(d1,d2,d3)
   end
end


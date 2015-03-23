require 'nn'

local Crop, parent = torch.class('nn.Crop','nn.Module')

function Crop:__init(iH,iW,oH,oW)
   self.iH = iH
   self.iW = iW
   self.oH = oH
   self.oW = oW
   self.dH = math.floor((self.iH-self.oH)/2)+1
   self.dW = math.floor((self.iW-self.oW)/2)+1
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Crop:updateOutput(input)  
   local nSamples = input:size(1)
   local nMaps = input:size(2)
   self.output:resize(nSamples, nMaps, self.oH, self.oW)
   self.output:copy(input:narrow(3,self.dH,self.oH):narrow(4,self.dW, self.oW))
   return self.output
end

function Crop:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size())
   self.gradInput:zero()
   self.gradInput:narrow(3,self.dH,self.oH):narrow(4,self.dW,self.oW):copy(gradOutput)
   return self.gradInput
end



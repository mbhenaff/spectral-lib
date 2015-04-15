require 'nn'

local ZeroBorders, parent = torch.class('nn.ZeroBorders','nn.Module')

function ZeroBorders:__init(iH,iW,dH,dW)
   self.iH = iH
   self.iW = iW
   self.dH = dH
   self.dW = dW
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function ZeroBorders:updateOutput(input)  
   self.output:resize(input:size())
   self.output:copy(input)
   libspectralnet.crop_zeroborders(self.output, self.dH, self.dW)
   return self.output
end

function ZeroBorders:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size())
   self.gradInput:copy(gradOutput)
   libspectralnet.crop_zeroborders(self.gradInput, self.dH, self.dW)
   return self.gradInput
end



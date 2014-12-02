require 'nn'

local Crop, parent = torch.class('nn.Crop','nn.Module')

-- For a set of images of size [iH x iW], set the borders to zero. 
-- The size of the border on each side is specified by the rows/cols arguments. 
function Crop:__init(iH,iW,rows,cols)
   self.iH = iH
   self.iW = iW
   self.rows = rows
   self.cols = cols
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function Crop:updateOutput(input)   
   self.output:resize(input:size())
   self.output:copy(input)
   cucomplex.crop_zeroborders(self.output, self.rows, self.cols)
   return self.output
end

function Crop:updateGradInput(input, gradOutput)
   self.gradInput:resize(gradOutput:size())
   self.gradInput:copy(gradOutput)
   cucomplex.crop_zeroborders(self.gradInput, self.rows, self.cols)
   return self.gradInput
end








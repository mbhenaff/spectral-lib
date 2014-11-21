require 'nn'

local Crop, parent = torch.class('nn.Crop','nn.Module')

function Crop:__init(iH,iW,rows,cols,complex)
   self.complex = complex or false
   self.iH = iH
   self.iW = iW
   self.rows = rows
   self.cols = cols
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

-- assuming input is 4D
function Crop:updateOutput(input)
   if self.complex then
      self.output:resize(input:size(1),input:size(2),input:size(3)-2*self.rows,input:size(4)-2*self.cols,2)
      self.output:copy(input[{{},{},{self.rows+1,self.iH-self.rows},{self.cols+1,self.iW-self.cols},{}}])
   else
      self.output:resize(input:size(1),input:size(2),input:size(3)-2*self.rows,input:size(4)-2*self.cols)
      self.output:copy(input[{{},{},{self.rows+1,self.iH-self.rows},{self.cols+1,self.iW-self.cols}}])
   end

   return self.output
end

function Crop:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size())
   self.gradInput:zero()
   if self.complex then
      self.gradInput[{{},{},{self.rows+1,self.iH-self.rows},{self.cols+1,self.iW-self.cols},{}}]:copy(gradOutput)
   else
      self.gradInput[{{},{},{self.rows+1,self.iH-self.rows},{self.cols+1,self.iW-self.cols}}]:copy(gradOutput)
   end
   return self.gradInput
end




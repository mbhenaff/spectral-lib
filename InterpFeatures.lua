

require 'nn'


local InterpFeatures, parent = torch.class('nn.InterpFeatures', 'nn.Module')

function InterpFeatures:__init(iF, oF, interpType)
   parent.__init(self)
   self.iF = iF
   self.oF = oF
   self.kernel = interpKernel(iF,oF,interpType)
   self.kernelT = self.kernel:t():clone()
   self.bufferIn = torch.FloatTensor()
   self.bufferOut = torch.FloatTensor()
end


function InterpFeatures:featureTransform(input, M)
   local nSamples = input:size(1)
   local nInputPlanes = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   local nOutputPlanes = M:size(1)
   self.bufferIn:resize(input:size())
   self.bufferIn:copy(input)
   self.bufferIn = self.bufferIn:transpose(1,2):contiguous():resize(nInputPlanes,nSamples*iH*iW*2)
   self.bufferOut:resize(nOutputPlanes, nSamples*iH*iW*2)
   self.bufferOut:zero()   
   self.bufferOut:addmm(M, self.bufferIn)
   self.bufferOut:resize(nOutputPlanes,nSamples,iH,iW,2)
   self.bufferOut = self.bufferOut:transpose(1,2):contiguous()
   return self.bufferOut
end


-- assuming inputs are [nSamples x nFeatures x iH x iW]
function InterpFeatures:updateOutput(input)
   self.output = self:featureTransform(input, self.kernelT)
   return self.output
end

function InterpFeatures:updateGradInput(input, gradOutput)
   self.gradInput = self:featureTransform(gradOutput, self.kernel)
   return self.gradInput
end




require 'nn'

local GraphMaxPooling, parent = torch.class('nn.GraphMaxPooling', 'nn.Module')

function GraphMaxPooling:__init(clusters)
   parent.__init(self)
   self.clusters = clusters
   self.nClusters = clusters:size(1)
   self.poolSize = clusters:size(2)
   self.output = torch.Tensor(self.nClusters)
   self.indices = torch.Tensor(self.nClusters)
   self:reset()
end

function GraphMaxPooling:updateOutput(input)
   if input:nDimension() == 3 then
      self.output:resize(input:size(1), input:size(2), self.nClusters)
      self.indices:resize(input:size(1), input:size(2), self.nClusters)
   elseif input:nDimension() == 2 then
      self.output:resize(input:size(1), self.nClusters)
      self.indices:resize(input:size(1), self.nClusters)
   else 
      error('wrong number of dimensions')
   end
   self.output:zero()
   self.indices:zero()
   --fprop_cpu(input, self.output, self.clusters, self.indices)
   libspectralnet.graph_pool_fprop(input, self.output, self.clusters, self.indices) 
   return self.output
end

function GraphMaxPooling:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size())
   self.gradInput:zero()
   --bprop_cpu(self.gradInput, gradOutput, self.indices)
   libspectralnet.graph_pool_bprop(self.gradInput, gradOutput, self.indices)
   return self.gradInput
end



function fprop_cpu(input, output, clusters, indices)
   local nClusters = clusters:size(1)
   local poolsize = clusters:size(2)
   local c = torch.Tensor(poolsize)
   local indx = torch.Tensor(poolsize)
   for k = 1,input:size(1) do 
      for n = 1,input:size(2) do 
         for i = 1,nClusters do 
            for j = 1,poolsize do 
               c[j] = input[k][n][clusters[i][j]]
               indx[j] = clusters[i][j]
            end
            local s,ix = torch.sort(c,true)
            output[k][n][i] = s[1]
            indices[k][n][i] = indx[ix[1]]
         end
      end
   end
end

function bprop_cpu(gradInput, gradOutput, indices)
   local nClusters = gradOutput:size(3)
   gradInput:zero()
   for k = 1,gradInput:size(1) do 
      for n = 1,gradInput:size(2) do 
         for i = 1,nClusters do 
            local ix = indices[k][n][i]
            gradInput[k][n][ix] = gradInput[k][n][ix] + gradOutput[k][n][i]
         end
      end
   end
end
   
      

      
         
         
   




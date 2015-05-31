-- general datasource object for training graph conv networks.
-- data is loaded and presented in the form [nSamples x nChannels x dim]
require 'torch'
matio = require 'matio'
local Datasource = torch.class('Datasource')

function Datasource:__init(dataset,normalization,testTime)
   print(normalization)

   self.testTime = testTime or false
   self.alpha = 0.1
   -- load the datasets and format to be [nSamples x nChannels x dim]
   local normalization = normalization or 'none'
   print('Loading dataset')
   self.output = torch.Tensor()
   self.labels = torch.LongTensor()
   if dataset == 'cifar' then
      self.train_set = {}
      self.test_set = {}
      local cifar = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/cifar_10_norm.th')
      self.train_set.data = cifar.trdata
      self.test_set.data = cifar.tedata
      self.train_set.labels = cifar.trlabels
      self.test_set.labels = cifar.telabels
      self.nClasses = 10
      self.nChannels = 3
      self.dim = 32*32
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
   elseif dataset == 'mnist' then
      self.train_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/train_28x28.th7nn')
      self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/test_28x28.th7nn')
      self.nClasses = 10
      self.nChannels = 1
      self.dim = 28*28
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
   elseif dataset == 'reuters' then
      self.train_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/train.th')
      self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/test.th')
      self.nClasses = 50
      self.nChannels = 1
      self.dim = 2000
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
      normalization = 'log'
   elseif dataset == 'timit' then
      self.train_set = torch.load('/scratch/timit/' .. 'train' .. '/data_winsize_15.th')
      self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/torch_datasets/timit/' .. 'dev' .. '/data_winsize_15.th')
      self.train_set.labels = self.train_set.labels + 1
      self.test_set.labels = self.test_set.labels + 1
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
      self.nChannels = 15
      self.dim = 120
   elseif dataset == 'merck1' or dataset == 'merck6' then
      local num = string.match(dataset,"%d+")
      local nTrain
      if dataset == 'merck6' then
         self.dim = 6499
         nTrain = 37388
         nTest = 12406
      else 
         self.dim = 6559
         nTrain = 37241
         nTest = 12338
      end
      self.train_set = {}
      self.train_set.data = torch.Tensor(nTrain,self.dim):zero()
      self.train_set.labels = torch.Tensor(nTrain):zero()
      local cntr = 1
      for i = 1,5 do 
         local chunk = matio.load('/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/merck' .. num .. '_train_chunk' .. i .. '.mat')
         self.train_set.data[{{cntr,cntr+chunk.data:size(1)-1}}]:copy(chunk.data)
         self.train_set.labels[{{cntr,cntr+chunk.data:size(1)-1}}]:copy(chunk.labels)
         cntr = cntr + chunk.data:size(1)
      end
      
      if self.testTime then
         self.test_set = {}
         self.test_set.data = torch.Tensor(nTest,self.dim):zero()
         self.test_set.labels = torch.Tensor(nTest):zero()
         local cntr = 1
         for i = 1,5 do 
            local chunk = matio.load('/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/merck' .. num .. '_test_chunk' .. i .. '.mat')
            print(chunk)
            print(self.test_set.data[{{cntr,cntr+chunk.data:size(1)-1}}]:size())
            self.test_set.data[{{cntr,cntr+chunk.data:size(1)-1}}]:copy(chunk.data)
            self.test_set.labels[{{cntr,cntr+chunk.data:size(1)-1}}]:copy(chunk.labels)
            cntr = cntr + chunk.data:size(1)
         end
      else
         local x = self.train_set
         local nSamples = nTrain
         local nTrain = math.floor(nSamples*(1-self.alpha))
         local nTest = nSamples - nTrain
         torch.manualSeed(314)
         local perm = torch.randperm(nSamples)
         self.train_set2 = {}
         self.test_set = {}
         self.train_set2.data = torch.Tensor(nTrain,self.dim)
         self.test_set.data = torch.Tensor(nTest,self.dim)
         self.train_set2.labels = torch.Tensor(nTrain,1)
         self.test_set.labels = torch.Tensor(nTest,1)
         print(x)
         for i = 1,nTrain do 
            self.train_set2.data[i]:copy(x.data[perm[i]])
            self.train_set2.labels[i][1] = x.labels[perm[i]]
         end
         for i = 1,nTest do 
            self.test_set.data[i]:copy(x.data[perm[i+nTrain]])
            self.test_set.labels[i][1] = x.labels[perm[i+nTrain]]
         end
         self.train_set = self.train_set2
      end

      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
      self.nChannels = 1
      self.train_set.labels = self.train_set.labels:squeeze()
      self.test_set.labels = self.test_set.labels:squeeze()
      if normalization ~= 'whitening' then normalization = 'none' end

   elseif string.match(dataset,'merck') then
      local num = string.match(dataset,"%d+")
      if self.testTime then
         self.test_set = matio.load('/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/merck' .. num .. '_test.mat')
         self.train_set = matio.load('/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/merck' .. num .. '_train.mat')
         self.dim = self.test_set.data:size(2)
         self.train_set.data = self.train_set.data:contiguous()
         self.train_set.labels = self.train_set.labels:contiguous()
         self.test_set.data = self.test_set.data:contiguous()
         self.test_set.data = self.test_set.data:contiguous()

      else
         local x = matio.load('/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/merck' .. num .. '_train.mat')
         local nSamples = x.data:size(1)
         self.dim = x.data:size(2)
         local nTrain = math.floor(nSamples*(1-self.alpha))
         local nTest = nSamples - nTrain
         torch.manualSeed(314)
         local perm = torch.randperm(nSamples)
         self.train_set = {}
         self.test_set = {}
         self.train_set.data = torch.Tensor(nTrain,self.dim)
         self.test_set.data = torch.Tensor(nTest,self.dim)
         self.train_set.labels = torch.Tensor(nTrain,1)
         self.test_set.labels = torch.Tensor(nTest,1)
         for i = 1,nTrain do 
            self.train_set.data[i]:copy(x.data[perm[i]])
            self.train_set.labels[i]:copy(x.labels[perm[i]])
         end
         for i = 1,nTest do 
            self.test_set.data[i]:copy(x.data[perm[i+nTrain]])
            self.test_set.labels[i]:copy(x.labels[perm[i+nTrain]])
         end
      end
      if normalization ~= 'whitening' then normalization = 'none' end
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
      self.nChannels = 1
      self.train_set.labels = self.train_set.labels:squeeze()
      self.test_set.labels = self.test_set.labels:squeeze()
   end
   self.train_set.data:resize(self.nSamples.train,self.nChannels,self.dim)
   self.test_set.data:resize(self.nSamples.test,self.nChannels,self.dim)

   self.train_set.data = self.train_set.data:float()
   self.test_set.data = self.test_set.data:float()

   print(normalization)
   -- apply normalization
   if normalization == 'global' then
      self.mean = self.train_set.data:mean()
      self.train_set.data:add(-self.mean)
      self.test_set.data:add(-self.mean)
      self.std = self.train_set.data:std()
      self.train_set.data:div(self.std)
      self.test_set.data:div(self.std)
   elseif normalization == 'whitening' then
      require 'unsup'
      print('whitening data')
      self.train_set.data:resize(self.nSamples.train,self.nChannels*self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels*self.dim)
      self.mean = self.train_set.data:mean(1)
--      self.train_set.data:add(-1,self.mean:expandAs(self.train_set.data))
--      self.test_set.data:add(-1,self.mean:expandAs(self.test_set.data))
      self.train_set.data, means, P = unsup.zca_whiten(self.train_set.data)
      means:resize(1,self.dim)
      self.test_set.data:add(-1,means:expandAs(self.test_set.data))
      self.test_set.data = torch.mm(self.test_set.data,P:float())
--      U,S,V = torch.svd(self.train_set.data)
--      print(torch.min(S),torch.max(S),torch.mean(S))
--      self.train_set.data = self.train_set.data*V
--      self.test_set.data = self.test_set.data*V
--      print(S:size())
--      gnuplot.hist(S)

--      for i = 1,S:nElement() do if S[i] < 1e-4 then S[i] = 1e-4 end end
--      Sinv = torch.cdiv(torch.ones(S:nElement()):float(),S)
--      Sinv:resize(1,Sinv:nElement())
      

--      self.train_set.data = self.train_set.data:cmul(Sinv:expandAs(self.train_set.data))
--      self.test_set.data = self.test_set.data:cmul(Sinv:expandAs(self.test_set.data))
      
      if false then
         self.std = self.train_set.data:std(1)
         self.train_set.data:cdiv(self.std:expandAs(self.train_set.data))
         self.test_set.data:cdiv(self.std:expandAs(self.test_set.data))
         self.train_set.data:resize(self.nSamples.train,self.nChannels,self.dim)
         self.test_set.data:resize(self.nSamples.test,self.nChannels,self.dim)
      end
      cov = self.train_set.data:t()*self.train_set.data
      gnuplot.imagesc(cov)
   elseif  normalization == 'feature' then
      self.train_set.data:resize(self.nSamples.train,self.nChannels*self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels*self.dim)
      self.mean = self.train_set.data:mean(1)
      self.train_set.data:add(-1,self.mean:expandAs(self.train_set.data))
      self.test_set.data:add(-1,self.mean:expandAs(self.test_set.data))
      self.std = self.train_set.data:std(1)
      self.train_set.data:cdiv(self.std:expandAs(self.train_set.data))
      self.test_set.data:cdiv(self.std:expandAs(self.test_set.data))
      self.train_set.data:resize(self.nSamples.train,self.nChannels,self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels,self.dim)
   elseif normalization == 'sample' then
      self.train_set.data:resize(self.nSamples.train,self.nChannels*self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels*self.dim)
      local std = self.train_set.data:norm(2,2)
      self.train_set.data:cdiv(std:expandAs(self.train_set.data))
      local std = self.test_set.data:norm(2,2)
      self.test_set.data:cdiv(std:expandAs(self.test_set.data))
      self.train_set.data:resize(self.nSamples.train,self.nChannels,self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels,self.dim)
   elseif normalization == 'log' then
      self.train_set.data:resize(self.nSamples.train,self.nChannels*self.dim)
      self.test_set.data:resize(self.nSamples.test,self.nChannels*self.dim)
      self.train_set.data:add(1):log()
      self.test_set.data:add(1):log()
   elseif normalization == 'none' then
      --do nothing
   else
      error('unrecognized normalization')
   end

   self.train_set.data = self.train_set.data:contiguous()
   self.train_set.labels = self.train_set.labels:contiguous()
   self.test_set.data = self.test_set.data:contiguous()
   self.test_set.data = self.test_set.data:contiguous()

   print('training set is ' .. self.nSamples['train'] .. ' x ' .. self.dim)
   print('testing set is ' .. self.nSamples['test'] .. ' x ' .. self.dim)
   self.indx = torch.randperm(self.nSamples.train)
end


function Datasource:shuffle()
   self.indx = torch.randperm(self.nSamples.train)
end

function Datasource:nextBatch(batchSize, set)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   else
      error('set must be [train|test]')
   end
   self.output:resize(batchSize, self.nChannels, self.dim)
   self.labels:resize(batchSize)
   for i = 1, batchSize do
      local idx = torch.random(this_set.data:size(1))
      self.output[i]:copy(this_set.data[idx])
      --TODO: more GPU friendly
      self.labels[i] = this_set.labels[idx]
   end
   return {self.output, self.labels}
end

function Datasource:nextIteratedBatchPerm(batchSize, set, idx, perm)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   else
      error('set must be [train|test]')
   end
   self.output:resize(batchSize, self.nChannels, self.dim)
   self.labels:resize(batchSize)
   for i = 1, batchSize do
      local idx1 = (idx-1)*batchSize+i
      if idx1 > perm:size(1) then
         return nil
      end
      local idx2 = perm[idx1]
      self.output[i]:copy(this_set.data[idx2])
      --TODO: more GPU friendly
      self.labels[i] = this_set.labels[idx2]
   end
   return {self.output, self.labels}
end

function Datasource:nextIteratedBatch(batchSize, set, idx)
   assert(idx > 0)
   local this_set
   if set == 'train' then
      this_set = self.train_set
   elseif set == 'test' then
      this_set = self.test_set
   else
      error('set must be [train|test]')
   end
   self.output:resize(batchSize, self.nChannels, self.dim)
   self.labels:resize(batchSize)

   if idx*batchSize > this_set.data:size(1) then
      return nil
   else
      self.output:copy(this_set.data[{{(idx-1)*batchSize+1,idx*batchSize}}])
      self.labels:copy(this_set.labels[{{(idx-1)*batchSize+1,idx*batchSize}}])
      return {self.output, self.labels}
   end
end

function Datasource:type(typ)
   --   self.train_set.data = self.train_set.data:type(typ)
   --   self.test_set.data = self.test_set.data:type(typ)
   self.output = self.output:type(typ)
   if typ == 'torch.CudaTensor' then
      self.train_set.labels = self.train_set.labels:type(typ)
      self.test_set.labels = self.test_set.labels:type(typ)
      self.labels = self.labels:type(typ)
   else
      self.train_set.labels = self.train_set.labels:type('torch.LongTensor')
      self.test_set.labels = self.test_set.labels:type('torch.LongTensor')
      self.labels = self.labels:type('torch.LongTensor')
   end
end
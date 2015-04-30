-- general datasource object for training graph conv networks.
-- data is loaded and presented in the form [nSamples x nChannels x dim]
require 'torch'
local Datasource = torch.class('Datasource')

function Datasource:__init(dataset,normalization)
   -- load the datasets and format to be [nSamples x nChannels x dim]
   print('Loading dataset')
   self.output = torch.Tensor()
   self.labels = torch.LongTensor()
   if dataset == 'cifar' then
      local cifar = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/cifar_10_norm.th')
      self.train_set.data = cifar.trdata
      self.test_set.data = cifar.tedata
      self.train_set.labels = cifar.trlabels
      self.test_set.labels = cifar.telabels
      self.nClasses = 10
      self.nChannels = 3
      self.dim = 32*32
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
   elseif dataset == 'timit' then
      self.train_set = torch.load('/scratch/timit/' .. split .. '/data_winsize_15.th')
      self.test_set = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/torch_datasets/timit/' .. split .. '/data_winsize_15.th')
      self.train_set.labels = self.train_set.labels + 1
      self.test_set.labels = self.test_set.labels + 1
      self.nSamples = {['train'] = self.train_set.data:size(1),['test'] = self.test_set.data:size(1)}
      self.nChannels = 15
      self.dim = 120
   end
   self.train_set.data:resize(self.nSamples.train,self.nChannels,self.dim)
   self.test_set.data:resize(self.nSamples.test,self.nChannels,self.dim)

   self.train_set.data = self.train_set.data:float()
   self.test_set.data = self.test_set.data:float()

   -- apply normalization
   if normalization == 'global' then
      self.mean = self.train_set.data:mean()
      self.train_set.data:add(-self.mean)
      self.test_set.data:add(-self.mean)
      self.std = self.train_set.data:std()
      self.train_set.data:div(self.std)
      self.test_set.data:div(self.std)
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
   elseif normalization == 'none' then
      --do nothing
   end

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
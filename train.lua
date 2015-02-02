dofile('params.lua')

if opt.model == 'gconv1' or opt.model == 'gconv2' then
   poolneighbs = opt.poolsize
   L = torch.load('mresgraph/' .. opt.dataset .. '_laplacian_poolsize_' .. opt.poolsize .. '_stride_' .. opt.poolstride .. '_neighbs_' .. poolneighbs .. '.th')
 
   --L = torch.load('mresgraph/mnist_laplacian_spatialsim_poolsize_9_stride_4_neighbs_9.th')
   --L = torch.load('mresgraph/mnist_laplacian_spatialsim_poolsize_4_stride_4_neighbs_4.th')
   --L = torch.load('mresgraph/mnist_laplacian_poolsize_4_stride_4.th')
   V1 = L.V[1]:float()
   V2 = L.V[2]:float()
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
else
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
end

if trdata:nDimension() == 4 then 
   trdata:resize(trdata:size(1),trdata:size(2)*trdata:size(3)*trdata:size(4))
   tedata:resize(tedata:size(1),tedata:size(2)*tedata:size(3)*tedata:size(4))
end

nsamples = trdata:size(1)
dim = trdata:size(2)

if opt.dataset == 'reuters' then
   nclasses = 50
else
   nclasses = 10
end
torch.manualSeed(314)
model = nn.Sequential()
if opt.model == 'linear' then
   model:add(nn.Linear(dim, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()
elseif opt.model == 'mlp' then
   model:add(nn.Linear(dim, opt.nhidden))
   model:add(nn.Threshold())
   for i = 1,opt.nlayers-1 do 
      model:add(nn.Linear(opt.nhidden, opt.nhidden))
      model:add(nn.Threshold())
   end
   model:add(nn.Linear(opt.nhidden, nclasses))
elseif opt.model == 'gconv1' then
   local poolsize = 4
   -- scale GFT matrices
   print('V1 norm = ' .. estimate_norm(V1))
   local s1 = math.sqrt(dim)
   --V1:mul(s1)
   print('V1 norm = ' .. estimate_norm(V1))
   n1 = V1:size(1)
   if opt.rfreq > 0 then
      print(opt.rfreq)
      V1[{{},{math.floor(n1/opt.rfreq)+1,n1}}]:zero()
   end

   model:add(nn.Reshape(1,dim))
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k, V1))
   model:add(nn.Bias(opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.GraphMaxPooling(L.pools[1]:t():clone()))

   -- classifier layer
   model:add(nn.Reshape(opt.nhidden*dim/(poolsize)))
   model:add(nn.Linear(opt.nhidden*dim/(poolsize), nclasses))
   model:add(nn.LogSoftMax())

   -- initialize biases appropriately
   model:get(3):reset(1./math.sqrt(opt.k))

   -- send to GPU and reset pointers
   model = model:cuda()
   model:get(2):resetPointers()
elseif opt.model == 'gconv2' then
   local poolsize = opt.poolsize
   -- scale GFT matrices
   --local s1 = math.sqrt(math.sqrt(dim))
   --local s2 = math.sqrt(math.sqrt(dim/poolsize))
   print('V1 norm = ' .. estimate_norm(V1))
   print('V2 norm = ' .. estimate_norm(V2))
   local s1 = math.sqrt(dim)
   local s2 = math.sqrt(dim/poolsize)
   --V1:mul(s1)
   --V2:mul(s2)
   print('V1 norm = ' .. estimate_norm(V1))
   print('V2 norm = ' .. estimate_norm(V2))
   n1 = V1:size(1)
   n2 = V2:size(1)   
   if opt.rfreq > 0 then
      print(opt.rfreq)
      V1[{{},{math.floor(n1/opt.rfreq)+1,n1}}]:zero()
      V2[{{},{math.floor(n2/opt.rfreq)+1,n2}}]:zero()
   end

   model:add(nn.Reshape(1,dim))
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k, V1))
   model:add(nn.Bias(opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.GraphMaxPooling(L.pools[1]:t():clone()))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolsize, opt.k, V2))
   model:add(nn.Bias(opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.GraphMaxPooling(L.pools[2]:t():clone()))

   -- classifier layer
   model:add(nn.Reshape(opt.nhidden*dim/(poolsize^2)))
   model:add(nn.Linear(opt.nhidden*dim/(poolsize^2), nclasses))
   model:add(nn.LogSoftMax())

   -- initialize biases appropriately
   model:get(3):reset(1./math.sqrt(opt.k))
   model:get(7):reset(1./math.sqrt(opt.nhidden*opt.k))

   -- send to GPU and reset pointers
   model = model:cuda()
   model:get(2):resetPointers()
   model:get(6):resetPointers()
   
elseif opt.model == 'gconvperm' then
   local poolsize = 2
   model:add(nn.Reshape(1,dim))
   model:add(nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k, L.V[1]:mul(1):t():clone()))
   model:add(nn.Reshape(opt.nhidden, dim, 1))
   model:add(nn.Bias(opt.nhidden))
   --model:add(nn.Identity())
   model:get(4):reset(1./math.sqrt(opt.k))
   model:add(nn.Reshape(opt.nhidden, dim)) 
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(poolsize,1,poolsize,1))
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolsize, opt.k, L.V[2]:mul(1):t():clone()))
   model:add(nn.Reshape(opt.nhidden, dim/poolsize, 1))
   model:add(nn.Bias(opt.nhidden))
   --model:add(nn.Identity())
   model:get(10):reset(1./math.sqrt(opt.nhidden*opt.k))
   model:add(nn.Reshape(opt.nhidden, dim/poolsize))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(poolsize,1,poolsize,1))
   model:add(nn.Reshape(opt.nhidden*dim/(poolsize^2)))
   model:add(nn.Linear(opt.nhidden*dim/(poolsize^2), nclasses))
elseif opt.model == 'debug' then
   local poolsize = 4
   model:add(nn.Reshape(1,dim))
   model:add(nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k, L.V1))
   model:add(nn.Reshape(opt.nhidden, dim, 1))
   model:add(nn.Bias(opt.nhidden))
   --model:add(nn.Identity())
   model:get(4):reset(1./math.sqrt(opt.k))
   model:add(nn.Reshape(opt.nhidden, dim)) 
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(poolsize,1,poolsize,1))
   model:add(nn.Reshape(opt.nhidden*dim/poolsize))
   model:add(nn.Linear(opt.nhidden*dim/poolsize, nclasses))
else
   error('unrecognized model')
end
--print(model:getParameters():norm())
cutorch.synchronize()
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   learningRateDecay = 1/trdata:size(1)
}

if opt.log then
   logFile = assert(io.open(logFileName,'w'))
   logFile:write(opt.modelFile)
end

-- these will record performance
trloss = torch.Tensor(opt.epochs)
teloss = torch.Tensor(opt.epochs)
tracc = torch.Tensor(opt.epochs)
teacc = torch.Tensor(opt.epochs)
--model:get(2):printNorms()

cutorch.synchronize()

for i = 1,opt.epochs do
   local trsize = trdata:size(1)
   inputs = torch.CudaTensor(opt.batchSize, dim):zero()
   targets = torch.CudaTensor(opt.batchSize)

   -- get model parameters
   w,dL_dw = model:getParameters()
cutorch.synchronize()
   --print(w:norm())
cutorch.synchronize()
   --print('# params: ' .. w:nElement())
   local shuffle = torch.randperm(trsize)
   -- train!
   for t = 1,(trsize-opt.batchSize),opt.batchSize do
      xlua.progress(t,trsize)
      -- create minibatch
      for i = 1,opt.batchSize do
         inputs[i]:copy(trdata[shuffle[t+i]])
         targets[i]=trlabels[shuffle[t+i]]
      end		      
      -- create closure to evaluate L(w) and dL/dw
      local feval = function(w_)
                       if w_ ~= w then
                          w:copy(w_)
                       end
                       dL_dw:zero()
                       local outputs = model:forward(inputs)
                       local L = criterion:forward(outputs,targets)
                       local dL_do = criterion:backward(outputs,targets)
                       model:backward(inputs,dL_do)
                       return L, dL_dw
                    end
      if opt.optim == 'sgd' then
         optim.sgd(feval,w,optimState)
      else
         optim.adagrad(feval, w, optimState)
      end
      --model:get(2):printNorms()
   end
   
   function computePerf(data,labels)
      local nSamples = data:size(1)
      local loss = 0
      local correct = 0
      for t = 1,(nSamples-opt.batchSize),opt.batchSize do
         xlua.progress(t,nSamples)
         inputs:copy(data[{{t,t+opt.batchSize-1},{}}])
         targets:copy(labels[{{t,t+opt.batchSize-1}}])
         local out = model:updateOutput(inputs)
         loss = loss + criterion:forward(out,targets)[1]
         for k = 1,opt.batchSize do
            local s,indx = torch.sort(out[k]:float(),true)
            local maxind = indx[1]
            if maxind == targets[k] then
               correct = correct + 1
            end
         end
      end
      local acc = correct / data:size(1)
      return round(loss/data:size(1),8), round(acc,6)
   end
   trainLoss, trainAcc = computePerf(trdata, trlabels)
   testLoss, testAcc = computePerf(tedata, telabels)
   local outString = 'Epoch ' .. i .. ' | trloss = ' .. trainLoss .. ', tracc = ' .. trainAcc .. ' | ' .. 'teloss = ' .. testLoss .. ', teacc=' .. testAcc .. '\n'
   print(outString)
   trloss[i] = trainLoss
   teloss[i] = testLoss
   tracc[i] = trainAcc
   teacc[i] = testAcc
   if opt.log then
      logFile:write(outString)
   end
end

if opt.log then
   logFile:close()
   torch.save(opt.savePath .. opt.modelFile .. '.model',{model=model,opt=opt,trloss=trloss,teloss=teloss,tracc=tracc,teacc=teacc})
end





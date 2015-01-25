dofile('params.lua')

if opt.dataset == 'imagenet' then
   trdata,trlabels = loadData(opt.dataset,'train1')
   tedata,telabels = loadData(opt.dataset,'test')
else
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
end

nChannels = trdata:size(2)
iH = trdata:size(3)
iW = trdata:size(4)

if opt.dataset == 'imagenet' then
   nclasses = 100
else
   nclasses = 10
end

model = nn.Sequential()
if opt.model == 'spatial1' then
   model:add(nn.SpatialConvolutionBatch(nChannels,opt.nhidden,opt.k,opt.k))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
   local d = math.floor((iH-opt.k+1)/2)*math.floor((iW-opt.k+1)/2)*opt.nhidden
   model:add(nn.Reshape(d))
   model:add(nn.Linear(d,nclasses))
elseif opt.model == 'spatial2' then
   pool = 2
   --model:add(nn.SpatialContrastiveNormalization(nChannels,image.gaussian(opt.k)))
   model:add(nn.SpatialConvolutionBatch(nChannels,opt.nhidden,opt.k,opt.k))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(pool,pool,pool,pool))
   local oH1 = math.floor((iH-opt.k+1)/pool)
   local oW1 = math.floor((iW-opt.k+1)/pool)
   local d = oH1*oW1*opt.nhidden
   --model:add(nn.SpatialContrastiveNormalization(opt.nhidden,image.gaussian(math.floor(opt.k/3))))
   model:add(nn.SpatialConvolutionBatch(opt.nhidden,opt.nhidden,opt.k,opt.k))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(pool,pool,pool,pool))   
   local oH2 = math.floor((oH1-opt.k+1)/pool)
   local oW2 = math.floor((oW1-opt.k+1)/pool)
   model:add(nn.Reshape(oH2*oW2*opt.nhidden))
   if opt.dropout then 
      model:add(nn.Dropout())
   end
   model:add(nn.Linear(oH2*oW2*opt.nhidden,nclasses))
elseif opt.model == 'spectral1' then
   opt.real = 'real'
   opt.kH = opt.k
   opt.kW = opt.k
   opt.ncrop = 0
   pool = 2
   model:add(nn.SpectralConvolutionImage(opt.batchSize,nChannels,opt.nhidden,iH,iW,opt.kH,opt.kW,opt.interp,opt.realKernels))
   model:add(nn.Real(opt.real))
   model:add(nn.Bias(opt.nhidden))
   model:get(3):reset(1./math.sqrt(nChannels*opt.kH*opt.kW))
   if opt.ncrop > 0 then
      model:add(nn.Crop(iH,iW,opt.ncrop,opt.ncrop))
   end   
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(pool,pool,pool,pool))

   model:add(nn.Reshape((iH/pool)*(iW/pool)*opt.nhidden))
   model:add(nn.Linear((iH/pool)*(iW/pool)*opt.nhidden,nclasses))
elseif opt.model == 'spectral2' then
   opt.real = 'real'
   opt.kH = opt.k
   opt.kW = opt.k
   opt.ncrop = 0
   pool = 2
   model:add(nn.SpectralConvolutionImage(opt.batchSize,nChannels,opt.nhidden,iH,iW,opt.kH,opt.kW,opt.interp,opt.realKernels))
   model:add(nn.Real(opt.real))
   model:add(nn.Bias(opt.nhidden))
   model:get(3):reset(1./math.sqrt(nChannels*opt.kH*opt.kW))
   if opt.ncrop > 0 then
      model:add(nn.Crop(iH,iW,opt.ncrop,opt.ncrop))
   end   
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(pool,pool,pool,pool))

   model:add(nn.SpectralConvolutionImage(opt.batchSize, opt.nhidden, opt.nhidden, iH/pool, iW/pool, opt.kH, opt.kW, opt.interp, opt.realKernels))
   model:add(nn.Real(opt.real))
   model:add(nn.Bias(opt.nhidden))
   model:get(3):reset(1./math.sqrt(opt.nhidden*opt.kH*opt.kW))
   if opt.ncrop > 0 then
      model:add(nn.Crop(iH,iW,opt.ncrop,opt.ncrop))
   end   
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(pool,pool,pool,pool))

   model:add(nn.Reshape((iH/pool^2)*(iW/pool^2)*opt.nhidden))
   model:add(nn.Linear((iH/pool^2)*(iW/pool^2)*opt.nhidden,nclasses))
end
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model = model:cuda()
model:reset()
criterion = criterion:cuda()

--model:get(1):printNorms()

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

chunk = 1
for i = 1,opt.epochs do
   if opt.dataset == 'imagenet' then
      print('loading imagenet chunk ' .. chunk)
      trdata,trlabels = loadData(opt.dataset,'train' .. chunk)
      chunk = chunk % 4 + 1
   end
      
   local trsize = trdata:size(1)
   inputs = torch.CudaTensor(opt.batchSize, trdata:size(2),trdata:size(3), trdata:size(4)):zero()
   targets = torch.CudaTensor(opt.batchSize)

   -- get model parameters
   w,dL_dw = model:getParameters()
   local shuffle = torch.randperm(trsize)
   -- train!
   for t = 1,(trsize-opt.batchSize),opt.batchSize do
      xlua.progress(t,trsize)
      -- create minibatch
      inputs:copy(trdata[{{t,t+opt.batchSize-1}}])
      targets:copy(trlabels[{{t,t+opt.batchSize-1}}])
      if false then
      for i = 1,opt.batchSize do 
         inputs[i]:add(-inputs[i]:mean())
         inputs[i]:mul(math.max(1/inputs[i]:std(),1e-4))
      end
      end

      if false then
         for i = 1,opt.batchSize do
            inputs[i]:copy(trdata[shuffle[t+i]])
            targets[i]=trlabels[shuffle[t+i]]
         end
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
      optim.sgd(feval,w,optimState)
   --local p1,g1 = model:get(1):getParameters()
      --model:get(1):printNorms()
   --print('grad norm = ' .. g1:norm())
      collectgarbage()
   end
   
   function computePerf(data,labels, verbose)
      local nSamples = data:size(1)
      local loss = 0
      local correct = 0
      setDropoutParam(model,0)
      for t = 1,(nSamples-opt.batchSize),opt.batchSize do
         xlua.progress(t,nSamples)
         inputs:copy(data[{{t,t+opt.batchSize-1}}])
         if false then
         for i = 1,inputs:size(1) do 
            inputs[i]:add(-inputs[i]:mean())
            inputs[i]:mul(math.max(1/inputs[i]:std(),1e-4))
         end
         end
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
         --collectgarbage()
      end
      local acc = correct / data:size(1)
      --return loss/data:size(1), acc
      setDropoutParam(model,0.5)
      return round(loss/data:size(1),8), round(acc,6)
   end
   trainLoss,trainAcc = computePerf(trdata, trlabels)
   testLoss,testAcc = computePerf(tedata, telabels)
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











dofile('params.lua')

trdata,trlabels = loadData(opt.dataset,'train')
tedata,telabels = loadData(opt.dataset,'test')

nChannels = trdata:size(2)
iH = trdata:size(3)
iW = trdata:size(4)
nclasses = 10

model = nn.Sequential()
if opt.conv == 'spatial' then
   model:add(nn.SpatialConvolutionBatch(nChannels,opt.nhidden,opt.kH,opt.kW))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
   local d = math.floor((iH-opt.kH+1)/2)*math.floor((iW-opt.kW+1)/2)*opt.nhidden
   model:add(nn.Reshape(d))
   model:add(nn.Linear(d,10))
elseif opt.conv == 'spectral' then
   model:add(nn.SpectralConvolution(opt.batchSize,nChannels,opt.nhidden,iH,iW,opt.kH,opt.kW,opt.interp,opt.realKernels))
   model:add(nn.Real(opt.real))
   if opt.ncrop > 0 then
      model:add(nn.Crop(iH,iW,opt.ncrop,opt.ncrop,false))
   end
   model:add(nn.Bias(opt.nhidden))
   --model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
   model:add(nn.Reshape(((iH-opt.ncrop*2)/2)*((iW-opt.ncrop*2)/2)*opt.nhidden))
   model:add(nn.Linear(((iH-opt.ncrop*2)/2)*((iW-opt.ncrop*2)/2)*opt.nhidden,10))
end
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model = model:cuda()
model:reset()
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

for i = 1,opt.epochs do
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
      optim.sgd(feval,w,optimState)
   end

   function computePerf(data,labels)
      local nSamples = data:size(1)
      local loss = 0
      local correct = 0
      for t = 1,(nSamples-opt.batchSize),opt.batchSize do
         xlua.progress(t,nSamples)
         inputs:copy(data[{{t,t+opt.batchSize-1},{},{},{}}])
         targets:copy(labels[{{t,t+opt.batchSize-1}}])
         local out = model:updateOutput(inputs)
         loss = loss + criterion:forward(out,targets)[1]
         for k = 1,opt.batchSize do
            local maxval = torch.max(out[k]:float())
            local maxind 
            for j = 1,10 do
               if out[k][j] == maxval then
                  maxind = j
               end
            end
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
   torch.save(opt.savePath .. opt.modelFile .. '.model',{model=model:float(),opt=opt,trloss=trloss,teloss=teloss,tracc=tracc,teacc=teacc})
end











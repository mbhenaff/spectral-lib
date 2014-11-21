dofile('params.lua')

if opt.type == 'spectral' then
   d,trlabels = loadData('cifar','train')
   trdata = torch.Tensor(d:size(1),4,32,32)
   trdata[{{},{1,3},{},{}}]:copy(d)
   d,telabels = loadData('cifar','test')
   tedata = torch.Tensor(d:size(1),4,32,32)
   tedata[{{},{1,3},{},{}}]:copy(d)
else
   trdata,trlabels = loadData('cifar','train')
   tedata,telabels = loadData('cifar','test')
end
iH = trdata:size(3)
iW = trdata:size(4)
nclasses = 10

model = nn.Sequential()
if opt.type == 'spatial' then
   model:add(nn.SpatialConvolutionBatch(3,opt.nhidden,opt.kH,opt.kW))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
   local d = math.floor((iH-opt.kH+1)/2)*math.floor((iW-opt.kW+1)/2)*opt.nhidden
   model:add(nn.Reshape(d))
   model:add(nn.Linear(d,10))
elseif opt.type == 'spectral' then
   model:add(nn.SpectralConvolution(opt.batchSize,4,opt.nhidden,32,32,opt.kH,opt.kW,opt.interp))
   if opt.ncrop > 0 then
      model:add(nn.Crop(32,32,opt.ncrop,opt.ncrop,true))
   end
   model:add(nn.Modulus())
   model:add(nn.Bias(opt.nhidden))
   --model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2,2,2,2))
   model:add(nn.Reshape(((iH-opt.ncrop*2)/2)*((iW-opt.ncrop*2)/2)*opt.nhidden))
   model:add(nn.Linear(((iH-opt.ncrop*2)/2)*((iW-opt.ncrop*2)/2)*opt.nhidden,10))
end
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model = model:cuda()
criterion = criterion:cuda()

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   learningRateDecay = 1/trdata:size(1)
}

logFile = assert(io.open(logFileName,'w'))
logFile:write(opt.modelFile)

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
      return round(loss/data:size(1),4), round(acc,4)
   end
   trainLoss, trainAcc = computePerf(trdata, trlabels)
   testLoss, testAcc = computePerf(tedata, telabels)
   local outString = 'Epoch ' .. i .. ' | trloss = ' .. trainLoss .. ', tracc = ' .. trainAcc .. ' | ' .. 'teloss = ' .. testLoss .. ', teacc=' .. testAcc .. '\n'
   print(outString)
   logFile:write(outString)
end

logFile:close()
torch.save(opt.savePath .. opt.modelFile .. '.model',{model=model,opt=opt})












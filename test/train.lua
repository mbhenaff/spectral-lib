dofile('params.lua')
dofile('data.lua')
dofile('model.lua')


optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   learningRateDecay = 1/trdata:size(1)
}

-- these will record performance
trloss = torch.Tensor(opt.epochs)
teloss = torch.Tensor(opt.epochs)
tracc = torch.Tensor(opt.epochs)
teacc = torch.Tensor(opt.epochs)

cutorch.synchronize()

for i = 1,opt.epochs do

   local trsize = trdata:size(1)
   inputs = torch.CudaTensor(opt.batchSize, nChannels,dim):zero()
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
         loss = loss + criterion:forward(out,targets)
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





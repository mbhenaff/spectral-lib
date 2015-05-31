dofile('params.lua')
dofile('data.lua')
dofile('model_merck.lua')

if opt.stop then 
   m1 = model:get(4)
   m2 = model:get(6)
   inp1 = torch.randn(opt.batchSize,opt.nhidden,dim):cuda()
   timer = torch.Timer()
   timer:reset()
   out1 = m1:forward(inp1)
   cutorch.synchronize()
   print(timer:time().real)
   timer:reset()
   out2 = m2:forward(out1)
   cutorch.synchronize()
   print(timer:time().real)
end


cutorch.synchronize()

for i = startEpoch,opt.epochs do
   -- get model parameters
   w,dL_dw = model:getParameters()
   local shuffle = torch.randperm(trsize)
   -- train!
   model:training()
   for t = 1,trsize/opt.batchSize + 1 do
      xlua.progress(t,math.floor(trsize/opt.batchSize + 1))
      local data 
         = datasource:nextIteratedBatchPerm(opt.batchSize, 'train', t, shuffle)
      if data == nil then break end
      local inputs, targets = unpack(data)
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
      elseif opt.optim == 'adagrad' then
         optim.adagrad(feval, w, optimState)
      end
   end
   
   function computePerf(set)
      model:evaluate()
      local loss, correct, nTotal, idx = 0, 0, 0, 1
      target = torch.Tensor(datasource.nSamples[set]):zero()
      pred = torch.Tensor(datasource.nSamples[set]):zero()
      while true do
         local data = datasource:nextIteratedBatch(opt.batchSize, set, idx)
         if data == nil then break end
         local inputs, targets = unpack(data)
         local outputs = model:updateOutput(inputs)
         target[{{nTotal+1,nTotal+opt.batchSize}}]:copy(targets)
         pred[{{nTotal+1,nTotal+opt.batchSize}}]:copy(outputs)
         loss = loss + criterion:forward(outputs,targets)
         nTotal = nTotal + opt.batchSize
         idx = idx + 1
      end
--      print(nTotal)
--      print(target:size())

      target = target[{{1,nTotal}}]
      pred = pred[{{1,nTotal}}]
      target:add(-torch.mean(target))
      pred:add(-torch.mean(pred))
      r = torch.sum(torch.cmul(target,pred))^2/((torch.cmul(target,target):sum())*(torch.cmul(pred,pred):sum()))
      model:training()
      return round(loss/nTotal,8),r
   end
   trainLoss,rtrain = computePerf('train')
   testLoss,rtest = computePerf('test')
--   gnuplot.plot(target,pred,'.')
   local outString = 'Epoch ' .. i .. ' | trloss = ' .. trainLoss .. ', r = ' .. rtrain .. ' | ' .. 'teloss = ' .. testLoss .. ', r = ' .. rtest .. '\n'
   print(outString)
   trrsqu[i] = rtrain
   tersqu[i] = rtest
   trloss[i] = trainLoss
   teloss[i] = testLoss
   if opt.log then
      logFile:write(outString)
   end
   collectgarbage()
end

if opt.log then
   logFile:close()
   torch.save(opt.savePath .. opt.modelFile .. '.model',{model=model,opt=opt,trloss=trloss,teloss=teloss,trrsqu=trrsqu,tersqu=tersqu,optimState=optimState})
end
trainLoss,rtrain = computePerf('train')

gnuplot.plot({'train',trloss},{'test',teloss})
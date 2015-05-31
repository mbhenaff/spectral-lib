

dataset_dict = {}
dataset_dict['merck1'] = '3A4'
dataset_dict['merck2'] = 'CB1'
dataset_dict['merck3'] = 'DPP4'
dataset_dict['merck4'] = 'HIVINT'
dataset_dict['merck5'] = 'HIVPROT'
dataset_dict['merck6'] = 'LOGD'
dataset_dict['merck7'] = 'METAB'
dataset_dict['merck8'] = 'NK1'
dataset_dict['merck9'] = 'OX1'
dataset_dict['merck10'] = 'OX2'
dataset_dict['merck11'] = 'PGP'
dataset_dict['merck12'] = 'PPB'
dataset_dict['merck13'] = 'RAT_F'
dataset_dict['merck14'] = 'TDI'
dataset_dict['merck15'] = 'THROMBIN'


function loadGraph2layer()
   if opt.graphscale == 'global' then
      local x = matio.load(opt.graphs_path .. 'alpha_' .. opt.alpha 
                           .. '/' .. opt.graph .. '_laplacian_gauss'
                              .. '_pool1_' .. opt.pool 
                              .. '_stride1_' .. opt.stride  
                              .. '_pool2_' .. opt.pool 
                              .. '_stride2_' .. opt.stride  
                              .. '.mat')
      V1 = x.V[1]:clone()
      V2 = x.V[2]:clone()
      pools1 = x.pools[1]:clone()
      pools2 = x.pools[2]:clone()
      print(x)
   elseif opt.graphscale == 'local' then
      local x = matio.load(opt.graphs_path .. 'alpha_' .. opt.alpha 
                           .. '/' .. opt.graph .. '_laplacian_gausslocal'
                              .. '_pool1_' .. opt.pool 
                              .. '_stride1_' .. opt.stride  
                              .. '_pool2_' .. opt.pool 
                              .. '_stride2_' .. opt.stride  
                              .. '.mat')
      V1 = x.V[1]:clone()
      V2 = x.V[2]:clone()
      pools1 = x.pools[1]:clone()
      pools2 = x.pools[2]:clone()
      print(x)
   end
   return V1,V2,pools1,pools2
end


if paths.filep(opt.savePath .. opt.modelFile .. '.model') then
   print('loading model from ' .. opt.savePath .. opt.modelFile .. '.model')
   local m = torch.load(opt.savePath .. opt.modelFile .. '.model')
   model = m.model
   optimState = m.optimState
   trloss = m.trloss
   teloss = m.teloss
   trrsqu = m.trrsqu
   tersqu = m.tersqu
   startEpoch = m.trloss:nElement() 
   trloss = torch.Tensor(opt.epochs)
   teloss = torch.Tensor(opt.epochs)
   trrsqu = torch.Tensor(opt.epochs)
   tersqu = torch.Tensor(opt.epochs)
   print(startEpoch)
   trloss[{{1,startEpoch}}]:copy(m.trloss[{{1,startEpoch}}])
   teloss[{{1,startEpoch}}]:copy(m.teloss[{{1,startEpoch}}])
   trrsqu[{{1,startEpoch}}]:copy(m.trrsqu[{{1,startEpoch}}])
   tersqu[{{1,startEpoch}}]:copy(m.tersqu[{{1,startEpoch}}])
   startEpoch = startEpoch + 1
else
   print('creating new model')
   startEpoch = 1
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 0--1/trsize
   }

   -- these will record performance
   trloss = torch.Tensor(opt.epochs)
   teloss = torch.Tensor(opt.epochs)
   trrsqu = torch.Tensor(opt.epochs)
   tersqu = torch.Tensor(opt.epochs)


   model = nn.Sequential()
   if opt.model == 'fc3' then
      model:add(nn.View(dim*nChannels))
      model:add(nn.Linear(dim*nChannels, opt.nhidden))
      model:add(nn.Threshold())
      model:add(nn.Linear(opt.nhidden, opt.nhidden))
      model:add(nn.Threshold())
      model:add(nn.Linear(opt.nhidden, 1))
      model = model:cuda()

   elseif opt.model == 'linear' then
      model:add(nn.View(dim*nChannels))
      model:add(nn.Linear(dim*nChannels, 1))
      model = model:cuda()

   elseif opt.model == 'dnn2' then
      model:add(nn.View(dim*nChannels))
      model:add(nn.Linear(dim*nChannels, 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000,500))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(500, 1))
      model = model:cuda()

   elseif opt.model == 'dnn4' then
      model:add(nn.View(dim*nChannels))
      --   model:add(nn.Dropout(0.2))
      model:add(nn.Linear(dim*nChannels, 4000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(4000,2000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(2000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000, 1))
      model = model:cuda()

   elseif opt.model == 'dnn4-wider' then
      model:add(nn.View(dim*nChannels))
      --   model:add(nn.Dropout(0.2))
      model:add(nn.Linear(dim*nChannels, 5000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(5000,2000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(2000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000, 1))
      model = model:cuda()



   elseif opt.model == 'gconv3-pool' then

      V1,_,pools1 = loadGraph2layer()

      --   model:add(nn.View(dim*nChannels))
      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 3
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      print(pools1:size())
      model:add(nn.GraphMaxPooling(pools1:t():clone()))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools1:size(2)))
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(opt.nhidden*pools1:size(2), 1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()


   elseif opt.model == 'gconv3' then

      V1,_,pools1 = loadGraph2layer()

      --   model:add(nn.View(dim*nChannels))
      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 3
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())


      -- classifier layer
      model:add(nn.View(opt.nhidden*dim))
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(opt.nhidden*dim, 1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()




   elseif opt.model == 'gconv2-pool' then

      V1,_,pools1 = loadGraph2layer()

      --   model:add(nn.View(dim*nChannels))
      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      print(pools1:size())
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools1:size(2)))
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(opt.nhidden*pools1:size(2), 1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   elseif opt.model == 'gconv2-pool-fc' then

      V1,_,pools1 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      print(pools1:size())
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools1:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools1:size(2), 100))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(100,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   elseif opt.model == 'gconv2-pool-gconv-fc' then

      V1,_,pools1 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      print(pools1:size())
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools1:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools1:size(2), 100))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(100,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()


   elseif opt.model == 'gconv2-pool-fc2' then

      V1,_,pools1 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      print(pools1:size())
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools1:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools1:size(2), 100))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(100, 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   elseif opt.model == 'gconv-pool-gconv-pool-fc' then

      V1,V2,pools1,pools2 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools2:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools2:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools2:size(2), 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   elseif opt.model == 'gconv-pool-gconv-pool-fc-fc' then

      V1, V2, pools1, pools2 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools2:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools2:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools2:size(2), 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000, 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()



   elseif opt.model == 'gconv-pool-dnn4' then
      V1,_,pools1 = loadGraph2layer()

      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))


      --   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, 1, dim, opt.k, V1, opt.interpScale))
      --   model:add(nn.SpectralConvolution(opt.batchSize, 4, 1, dim, opt.k, V1, opt.interpScale))


      model:add(nn.View(dim*nChannels))
      --   model:add(nn.Dropout(0.2))
      model:add(nn.Linear(dim*nChannels, 4000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(4000,2000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(2000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000,1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000, 1))


      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   elseif opt.model == 'gconv-deepreduce' then
      V1,V2,pools1,pools2 = loadGraph2layer()

      nhidden1 = opt.nhidden
      nhidden2 = opt.nhidden/2
      nhidden3 = opt.nhidden/4
      
      --   model:add(nn.View(dim*nChannels))
      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, nhidden1, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, nhidden1, nhidden2, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- conv layer 3
      model:add(nn.SpectralConvolution(opt.batchSize, nhidden2, nhidden3, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools2:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(nhidden3*pools2:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(nhidden3*pools2:size(2), 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()


   elseif opt.model == 'gconv-pool-gconv-pool-fc2' then

      V1,V2,pools1,pools2 = loadGraph2layer()

      --   model:add(nn.View(dim*nChannels))
      model:add(nn.View(nChannels, dim))
      -- conv layer 1
      model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

      -- conv layer 2
      model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
      model:add(nn.Threshold())

      -- pooling layer
      model:add(nn.GraphPooling(pools2:t():clone(), opt.pooltype))

      -- classifier layer
      model:add(nn.View(opt.nhidden*pools2:size(2)))
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(opt.nhidden*pools2:size(2), 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.25))
      model:add(nn.Linear(1000, 1000))
      model:add(nn.Threshold())
      model:add(nn.Dropout(0.1))
      model:add(nn.Linear(1000,1))

      -- send to GPU and reset pointers
      model = model:cuda()
      model:reset()

   else 
      error('unrecognized model')
   end

end

w = model:getParameters()
print('#params: ' .. w:nElement())

criterion = nn.MSECriterion()
criterion = criterion:cuda()

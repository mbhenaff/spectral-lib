-------------------------------
-- CREATE MODEL
-------------------------------
matio = require 'matio'

-- load GFT matrices and connection table
function loadGraph1layer()
   local x = matio.load(opt.graphs_path .. 'alpha_' .. opt.alpha 
                        .. '/reuters_laplacian_gauss.mat')
   connTable = x.NN
   return connTable
end



function loadGraph2layer()
   if opt.graphscale == 'global' then
      local x = matio.load(opt.graphs_path .. 'alpha_' .. opt.alpha 
                           .. '/' .. opt.graph .. '_laplacian_gauss'
                              .. '_pool1_' .. opt.pool 
                              .. '_stride1_' .. opt.stride  
                              .. '_pool2_' .. opt.pool 
                              .. '_stride2_' .. opt.stride  
                              .. 'rank_1.mat')
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

torch.manualSeed(314)
nInputs = dim*nChannels

-- make model
model = nn.Sequential()

function regularize() end

if opt.model == 'linear' then
   model:add(nn.View(dim*nChannels))
   model:add(nn.Linear(dim*nChannels, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'lc2' then
   model:add(nn.View(dim*nChannels))
   model:add(nn.LocallyConnected(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.LocallyConnected(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'lc3' then

   local nInputs = dim*nChannels
   model:add(nn.View(nInputs))
   
   model:add(nn.LocallyConnected(nInputs, nInputs, connTable))
   model:add(nn.Threshold())
   model:add(nn.LocallyConnected(nInputs, nInputs, connTable))
   model:add(nn.Threshold())
   model:add(nn.Linear(nInputs, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'lc3dropout' then

   local nInputs = dim*nChannels
   model:add(nn.View(nInputs))
   model:add(nn.Dropout(0.2))   
   model:add(nn.LocallyConnected(nInputs, nInputs, connTable))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.LocallyConnected(nInputs, nInputs, connTable))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(nInputs, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc-reuters' then 

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(dim*nChannels, 2000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(2000, 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()



elseif opt.model == 'fc-reuters-debug' then 
   V1,V2,pools1,pools2 = loadGraph2layer()
   model:add(nn.SpectralConvolution(opt.batchSize, 1, 2, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())
   --model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))
   model:add(nn.SpectralConvolution(opt.batchSize, 2, 1, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(dim*nChannels, 2000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(2000, 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()



elseif opt.model == 'fc2' then 
   model:add(nn.View(dim*nChannels))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc3' then 
   model:add(nn.View(dim*nChannels))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc3dropout' then 
   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'gc3' then 
   require 'regularization'


   L = x.L:cuda()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())

   model:get(3):initReg()
   model:get(6):initReg()
   function regularize(lambda)
      for i = 1,1 do
      model:get(3):regularize(L,lambda,opt.npts)
      model:get(6):regularize(L,lambda,opt.npts)
      end
   end
   model = model:cuda()

elseif opt.model == 'gc4' then 
   require 'regularization'

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())

   model:get(3):initReg()
   model:get(6):initReg()
   model:get(9):initReg()
   function regularize(lambda)
      model:get(3):regularize(L,lambda)
      model:get(6):regularize(L,lambda)
      model:get(9):regularize(L,lambda)
   end

   model = model:cuda()


elseif opt.model == 'fc4' then 
   model:add(nn.View(dim*nChannels))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc5' then 
   model:add(nn.View(dim*nChannels))
   model:add(nn.Linear(dim*nChannels, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'gconv1' then
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.Reshape(opt.nhidden*dim))
   model:add(nn.Linear(opt.nhidden*dim, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv2' then
   V1,V2,pools1,pools2 = loadGraph2layer()
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim))
   model:add(nn.Linear(opt.nhidden*dim, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv2-pool' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-pool-gconv' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-pool-gconv2' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-pool-gconv3' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 4
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-pool-gconv3-smallfc' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 4
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())


   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), 100))
   model:add(nn.Threshold())
   model:add(nn.Linear(100,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()



elseif opt.model == 'gconv2-pool-smallfc' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), 100))
   model:add(nn.Threshold())
   model:add(nn.Linear(100,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-pool-fc' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.View(nChannels, dim))
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*pools1:size(2), 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-pool-gconv-fc' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
   model:add(nn.View(nChannels, dim))
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, pools1:size(2), opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Linear(opt.nhidden*pools1:size(2), 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-pool-gconv-pool' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
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
--   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*pools2:size(2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-pool-gconv-pool-fc' then

   V1,V2,pools1,pools2 = loadGraph2layer()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
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
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*pools2:size(2), 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-deep1' then 

   poolsize1 = 8
   poolstride1 = 4
   poolsize2 = 8
   poolstride2 = 4
   V1,V2,pools1,pools2 = loadGraph2layer()

   model:add(nn.View(dim*nChannels))
   model:add(nn.Dropout(0.2))
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
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*pools2:size(2), 1000))
   model:add(nn.Threshold())
   model:add(nn.Dropout())
   model:add(nn.Linear(1000,nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-deep2' then 

   poolsize1 = 8
   poolstride1 = 4
   poolsize2 = 8
   poolstride2 = 4
   
   V1,V2,pools1,pools2 = loadGraph2layer()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools1:t():clone(), opt.pooltype))

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 4
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphPooling(pools2:t():clone(), opt.pooltype))

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim/(poolstride1*poolstride2)))
   model:add(nn.Linear(opt.nhidden*dim/(poolstride1*poolstride2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()
else
   error('unrecognized model')
end

cutorch.synchronize()
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

local nParams = model:getParameters():nElement()
print('#params: ' .. nParams)

if opt.log then
   logFile:write('\n#params: ' .. nParams .. '\n')
end
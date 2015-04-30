-------------------------------
-- CREATE MODEL
-------------------------------
matio = require 'matio'


-- load GFT matrices and connection table
if string.match(opt.model,'gconv') or string.match(opt.model,'lc') then
   graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   local x = matio.load(graphs_path .. 'alpha_' .. opt.alpha 
                        .. '/reuters_laplacian_poolsize_' 
                        .. 32 .. '_poolstride_' 
                        .. 16 .. '.mat')

   V1 = x.V1:clone():float()
   V2 = x.V1:clone():float()
end

if string.match(opt.model,'pool') then
   local graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   local x = matio.load(graphs_path .. 'alpha_' .. opt.alpha 
                        .. '/reuters_laplacian_poolsize_' 
                        .. opt.poolsize .. '_poolstride_' 
                        .. opt.poolstride .. '.mat')
   pools1 = x.pools[1]:clone()
   --pools2 = x.pools[2]:clone()
end


torch.manualSeed(314)
nInputs = dim*nChannels

-- make model
model = nn.Sequential()
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
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim))
   model:add(nn.Linear(opt.nhidden*dim, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv2-finalpool' then

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim/opt.poolstride))
   model:add(nn.Linear(opt.nhidden*dim/opt.poolstride, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv2-finalpool-dropout' then

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim/opt.poolstride))
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*dim/opt.poolstride, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv3-finalpool' then 
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim/opt.poolstride))
   model:add(nn.Linear(opt.nhidden*dim/opt.poolstride, nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

elseif opt.model == 'gconv-deep1' then 

   poolsize1 = 8
   poolstride1 = 4
   poolsize2 = 8
   poolstride2 = 4
   
   local graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   local x = matio.load(graphs_path .. '/reuters_laplacian_pool1_' .. poolsize1 .. '_stride1_' .. poolstride1 .. '_pool2_' .. poolsize2 .. '_stride2_' .. poolstride2 .. '.mat')


   V1 = x.V[1]:clone()
   V2 = x.V[2]:clone()
   pools1 = x.pools[1]:clone()
   pools2 = x.pools[2]:clone()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 4
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools2:t():clone()))

   -- classifier layer
   model:add(nn.View(opt.nhidden*dim/(poolstride1*poolstride2)))
   model:add(nn.Linear(opt.nhidden*dim/(poolstride1*poolstride2), nclasses))
   model:add(nn.LogSoftMax())

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()


elseif opt.model == 'gconv-deep2' then 

   poolsize1 = 8
   poolstride1 = 4
   poolsize2 = 8
   poolstride2 = 4
   
   local graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   local x = matio.load(graphs_path .. '/reuters_laplacian_pool1_' .. poolsize1 .. '_stride1_' .. poolstride1 .. '_pool2_' .. poolsize2 .. '_stride2_' .. poolstride2 .. '.mat')


   V1 = x.V[1]:clone()
   V2 = x.V[2]:clone()
   pools1 = x.pools[1]:clone()
   pools2 = x.pools[2]:clone()

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- conv layer 3
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 4
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolstride1, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   model:add(nn.GraphMaxPooling(pools2:t():clone()))

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
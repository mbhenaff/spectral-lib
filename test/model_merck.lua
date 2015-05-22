

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

elseif opt.model == 'gconv2-pool' then


   local graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   local x = matio.load(graphs_path .. 'alpha_' .. opt.alpha 
                        .. '/' .. opt.dataset .. '_laplacian_gauss_poolsize_' 
                        .. opt.poolsize .. '_poolstride_' 
                        .. opt.poolstride .. '.mat')
   pools1 = x.pools[1]:clone()
   print(x)

--   model:add(nn.View(dim*nChannels))
   model:add(nn.View(nChannels, dim))
   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1, opt.interpScale))
   model:add(nn.Threshold())

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim, opt.k, V2, opt.interpScale))
   model:add(nn.Threshold())

   -- pooling layer
   print(pools1:size())
   model:add(nn.GraphMaxPooling(pools1:t():clone()))

   -- classifier layer
   model:add(nn.View(opt.nhidden*pools1:size(2)))
   model:add(nn.Dropout())
   model:add(nn.Linear(opt.nhidden*pools1:size(2), 1))

   -- send to GPU and reset pointers
   model = model:cuda()
   model:reset()

else 
   error('unrecognized model')
end

w = model:getParameters()
print('#params: ' .. w:nElement())

criterion = nn.MSECriterion()
criterion = criterion:cuda()

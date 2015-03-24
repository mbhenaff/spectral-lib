-------------------------------
-- CREATE MODEL
-------------------------------


if opt.model == 'gconv1' or opt.model == 'gconv2' then
   local graphs_path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'
   --local graph_name = opt.dataset .. '_spatialsim_laplacian_poolsize_' .. opt.poolsize .. '_stride_' .. opt.poolstride .. '_neighbs_' .. opt.poolneighbs .. '.th') 
   local graph_name = 'timit_laplacian_120.th'
   L = torch.load(graphs_path .. graph_name)
--   V1 = L.V[1]:float()
--   V2 = L.V[2]:float()
   V1 = L:clone()
   V2 = L:clone()
   --V1 = torch.load('mresgraph/dct_kernel_32.th'):t():float()
   --V2 = torch.load('mresgraph/dct_kernel_16.th'):t():float()

end


torch.manualSeed(314)
model = nn.Sequential()
if opt.model == 'linear' then
   model:add(nn.Linear(dim, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc2' then 
   model:add(nn.Linear(dim, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc3' then 
   model:add(nn.Linear(dim, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc4' then 
   model:add(nn.Linear(dim, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, opt.nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(opt.nhidden, nclasses))
   model:add(nn.LogSoftMax())
   model = model:cuda()

elseif opt.model == 'fc5' then 
   model:add(nn.Linear(dim, opt.nhidden))
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
   local poolsize = 1
   -- scale GFT matrices
   print('V1 norm = ' .. estimate_norm(V1))
   local s1 = math.sqrt(dim)
   V1:mul(s1)
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
   --model:add(nn.GraphMaxPooling(L.pools[1]:t():clone()))

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
--   local poolsize = opt.poolsize
   local poolsize = 1
   -- scale GFT matrices
   print('V1 norm = ' .. estimate_norm(V1))
   print('V2 norm = ' .. estimate_norm(V2))
   n1 = V1:size(1)
   n2 = V2:size(1)   
   if opt.rfreq > 0 then
      print(opt.rfreq)
      V1[{{},{math.floor(n1/opt.rfreq)+1,n1}}]:zero()
      V2[{{},{math.floor(n2/opt.rfreq)+1,n2}}]:zero()
   end

   --model:add(nn.Reshape(1,dim))

   -- conv layer 1
   model:add(nn.SpectralConvolution(opt.batchSize, nChannels, opt.nhidden, dim, opt.k, V1))
   model:add(nn.Bias(opt.nhidden))
   model:add(nn.Threshold())
   --model:add(nn.GraphMaxPooling(L.pools[1]:t():clone()))

   -- conv layer 2
   model:add(nn.SpectralConvolution(opt.batchSize, opt.nhidden, opt.nhidden, dim/poolsize, opt.k, V2))
   model:add(nn.Bias(opt.nhidden))
   model:add(nn.Threshold())
   --model:add(nn.GraphMaxPooling(L.pools[2]:t():clone()))

   -- classifier layer
   model:add(nn.Reshape(opt.nhidden*dim/(poolsize^2)))
   model:add(nn.Linear(opt.nhidden*dim/(poolsize^2), nclasses))
   model:add(nn.LogSoftMax())

   -- initialize biases appropriately
   model:get(2):reset(1./math.sqrt(opt.k))
   model:get(5):reset(1./math.sqrt(opt.nhidden*opt.k))

   -- send to GPU and reset pointers
   model = model:cuda()
   model:get(1):resetPointers()
   model:get(4):resetPointers()   
else
   error('unrecognized model')
end

cutorch.synchronize()
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

print('#params: ' .. model:getParameters():nElement())
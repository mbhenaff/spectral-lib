-- test that spatial convnet and spectral with spatial kernel give same results

require 'cunn'
require 'spectralnet'
require 'loadData'

torch.setdefaulttensortype('torch.FloatTensor')

dataset = 'imagenet'
trdata,trlabels = loadData(dataset,'train1')
tedata,telabels = loadData(dataset,'test')
cutorch.setDevice(3)

trdata = torch.randn(100,3,128,128)

nChannels = trdata:size(2)
iH = trdata:size(3)
iW = trdata:size(4)
k = 5
kH = k
kW = k
nhidden = 48
batchSize = 32
pool = 2
twolayer = false
precision = 1e-3

if dataset == 'imagenet' then
   nclasses = 100
else
   nclasses = 10
end

function relativeErr(x1,x2)
   return torch.norm(x1:float()-x2:float())/math.min(x1:norm(),x2:norm())
end

--------------------------
-- spatial convnet model
--------------------------
model1 = nn.Sequential()
pad = (k-1)/2
model1:add(nn.SpatialConvolution(nChannels,nhidden,k,k,1,1,pad,pad))
model1:add(nn.ZeroBorders(iH,iW,pad,pad))
model1:add(nn.Threshold())
model1:add(nn.SpatialMaxPooling(pool,pool,pool,pool))
if twolayer then
   oH1 = math.floor((iH-k+1)/pool)
   oW1 = math.floor((iW-k+1)/pool)
   model1:add(nn.SpatialConvolution(nhidden,nhidden,k,k,1,1,pad,pad))
   model1:add(nn.ZeroBorders(oH1,oW1,pad,pad))
   model1:add(nn.Threshold())
   model1:add(nn.SpatialMaxPooling(pool,pool,pool,pool))   
end
model1 = model1:cuda()
model1:reset()

----------------------------------------------------------
-- spectral model with spatial kernel, real weights
----------------------------------------------------------
interp = 'spatial'
real = 'realpart'
model2 = nn.Sequential()
model2:add(nn.SpectralConvolutionImage(batchSize,nChannels,nhidden,iH,iW,kH,kW,interp,real))
model2:add(nn.Threshold())
model2:add(nn.SpatialMaxPooling(pool,pool,pool,pool))

if twolayer then
--   oH1 = math.floor((iH-k+1)/pool)
--   oW1 = math.floor((iW-k+1)/pool)
   oH1 = iH/pool
   oW1 = iW/pool
   model2:add(nn.SpectralConvolutionImage(batchSize, nhidden, nhidden, oH1, oW1, kH, kW, interp, real))
   model2:add(nn.Threshold())
   model2:add(nn.SpatialMaxPooling(pool,pool,pool,pool))
end
model2 = model2:cuda()
model2:reset()
----------------------------------------------------------
 
w1=model1:get(1).weight:clone()
b1=model1:get(1).bias:clone()
b1:zero()

torch.manualSeed(123)

model1:get(1).weight:copy(w1)
model1:get(1).bias:copy(b1)
model1:get(1).gradWeight:zero()
model2:get(1).weightPreimage:select(5,1):copy(w1)
model2:get(1).bias:copy(b1)
model2:get(1).gradWeightPreimage:zero()

if twolayer then
   w2=model1:get(5).weight:clone()
   b2=model1:get(5).bias:clone()
   b2:zero()
   w2:normal()
   b2:normal()
   model1:get(5).weight:copy(w2)
   model1:get(5).bias:copy(b2)
   model2:get(4).weightPreimage:select(5,1):copy(w2)
   model2:get(4).bias:copy(b2)
end

for i = 1,5 do
   print('\n**************')
   print('Run ' .. i)
   print('**************')

   --model2:reset()
   if false then
      model1:get(1).gradWeight:zero()
      model2:get(1).gradWeightPreimage:zero()
      if twolayer then
         model1:get(5).gradWeight:zero()
         model2:get(4).gradWeightPreimage:zero()
      end
   else
      model1:zeroGradParameters()
      model2:zeroGradParameters()
   end

   inputs = trdata[{{i,i+batchSize-1}}]:cuda()
   labels = trlabels[{{i,i+batchSize-1}}]

   out1 = model1:forward(inputs:clone())
   out2 = model2:forward(inputs:clone())
   if false then
   g1 = out1:clone()
   g2 = out2:clone()
   model1:updateGradInput(inputs, g1)
   model2:updateGradInput(inputs, g2)
   model1:backward(inputs, g1)
   model2:backward(inputs, g2)
   end

   -- check error on fprop
   err = relativeErr(model1:get(2).output:float() , model2:get(1).output:float())
   assert(err < precision, 'output error in 1st conv = ' .. err)

   err = relativeErr(model1:get(3).output:float() , model2:get(2).output:float())
   assert(err < precision, 'output error 1st Threshold = ' .. err)

   if twolayer then
      err = relativeErr(model1:get(6).output:float() , model2:get(4).output:float())
      print(model1)
      print(model2)
      assert(err < precision, 'output error 2nd conv = ' .. err)
      
      err = relativeErr(model1:get(7).output:float() , model2:get(5).output:float())
      assert(err < precision, 'output error 2nd Threshold = ' .. err)
   end
   print(model1.output:size())
   print(model2.output:size())
   print(model1)
   print(model2)
   err = relativeErr(model1.output:float() , model2.output:float())
   assert(err < precision, 'final error = ' .. err)
   print('no errors in outputs')

   -- check gradient error
   err = relativeErr(model1:get(1).gradInput:float() , model2:get(1).gradInput:float())
   assert(err < precision, 'gradInput error 1st conv = ' .. err)

   err = relativeErr(model1:get(3).gradInput:float() , model2:get(2).gradInput:float())
   assert(err < precision, 'gradInput error 1st Threshold = ' .. err)

   if twolayer then
      err = relativeErr(model1:get(4).gradInput:float() , model2:get(3).gradInput:float())
      assert(err < precision, 'gradInput error 2nd conv = ' .. err)
   end
   print('no errors in gradients')


   err = relativeErr(model1:get(1).weight:float() , model2:get(1).weightPreimage:select(5,1):float())
   assert(err < precision, 'weight error 1st conv = ' .. err)

   -- check weight error
   err = relativeErr(model1:get(1).gradWeight:float() , model2:get(1).gradWeightPreimage:select(5,1):float())
   assert(err < precision, 'gradWeight error in 1st conv = ' .. err)

   if twolayer then
      err = relativeErr(model1:get(5).weight:float() , model2:get(4).weightPreimage:select(5,1):float())
      assert(err < precision, 'weight error in 2nd conv = ' .. err)

      err = relativeErr(model1:get(5).gradWeight:float() , model2:get(4).gradWeightPreimage:select(5,1):float())
      assert(err < precision, 'gradWeight error in 2nd conv = ' .. err)
   end
   print('no errors on weights or gradWeights')
end















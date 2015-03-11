-- test that spatial convnet and spectral with spatial kernel give same results

dofile('params.lua')
require 'cunn'
require 'Crop'
require 'SpectralConvolutionImageAll'
dataset = 'cifar'
trdata,trlabels = loadData(dataset,'train')
tedata,telabels = loadData(dataset,'test')
cutorch.setDevice(3)
nChannels = trdata:size(2)
iH = trdata:size(3)
iW = trdata:size(4)
k = 5
kH = k
kW = k
nhidden = 16
batchSize = 32
twolayer = true

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
pool = 1
model1:add(nn.SpatialConvolutionBatch(nChannels,nhidden,k,k))
model1:add(nn.Threshold())
model1:add(nn.SpatialMaxPooling(pool,pool,pool,pool))
if twolayer then
   oH1 = math.floor((iH-k+1)/pool)
   oW1 = math.floor((iW-k+1)/pool)
   model1:add(nn.SpatialConvolutionBatch(nhidden,nhidden,k,k))
   model1:add(nn.Threshold())
   model1:add(nn.SpatialMaxPooling(pool,pool,pool,pool))   
end
model1 = model1:cuda()
model1:reset()

----------------------------------------------------------
-- spectral model with spatial kernel, real weights
----------------------------------------------------------
interp = 'spatial'
real = 'real'
realKernels = true
pool = 1
model2 = nn.Sequential()
model2:add(nn.SpectralConvolutionImage(batchSize,nChannels,nhidden,iH,iW,kH,kW,interp,realKernels))
model2:add(nn.Real(real))
model2:add(nn.Bias(nhidden))
model2:add(nn.Crop(iH,iW,iH-kH+1,iW-kW+1))
model2:add(nn.Threshold())
model2:add(nn.SpatialMaxPooling(pool,pool,pool,pool))

model2:get(3):reset(1./math.sqrt(nChannels*kH*kW))

if twolayer then
   oH1 = math.floor((iH-k+1)/pool)
   oW1 = math.floor((iW-k+1)/pool)
   model2:add(nn.SpectralConvolutionImage(batchSize, nhidden, nhidden, oH1, oW1, kH, kW, interp, realKernels))
   model2:add(nn.Real(real))
   model2:add(nn.Bias(nhidden))
   model2:add(nn.Crop(oH1,oW1,oH1-k+1,oW1-k+1))
   model2:add(nn.Threshold())
   model2:add(nn.SpatialMaxPooling(pool,pool,pool,pool))
   model2:get(9):reset(1./math.sqrt(nhidden*kH*kW))
end
model2 = model2:cuda()
model2:reset()
----------------------------------------------------------
 
w1=model1:get(1).weight:clone()
b1=model1:get(1).bias:clone()

torch.manualSeed(123)

model1:get(1).weight:copy(w1)
model1:get(1).bias:copy(b1)
model2:get(1).weightPreimage:select(5,1):copy(w1)
model2:get(3).bias:copy(b1)

if twolayer then
   w2=model1:get(4).weight:clone()
   b2=model1:get(4).bias:clone()
   w2:normal()
   b2:normal()
   model1:get(4).weight:copy(w2)
   model1:get(4).bias:copy(b2)
   model2:get(7).weightPreimage:select(5,1):copy(w2)
   model2:get(9).bias:copy(b2)
end

for i = 1,10 do 

   inputs = trdata[{{i,i+batchSize-1}}]:cuda()
   labels = trlabels[{{i,i+batchSize-1}}]

   out1 = model1:forward(inputs:clone())
   out2 = model2:forward(inputs:clone())
   g1 = out1:clone()
   g2 = out2:clone()
   model1:updateGradInput(inputs, g1)
   model2:updateGradInput(inputs, g2)
   model1:backward(inputs, g1)
   model2:backward(inputs, g2)
   spatial1 = model1:get(1)
   spectral1 = model2:get(1)

   -- check error on fprop
   print('output error 1st conv')
   print(relativeErr(spatial1.output:float() , model2:get(4).output:float()))
   print('output error 1st Threshold')
   print(relativeErr(model1:get(2).output:float() , model2:get(5).output:float()))

   if twolayer then
      print('output error 2nd conv')
      print(relativeErr(model1:get(4).output:float() , model2:get(10).output:float()))
      print('output error 2nd Threshold')
      print(relativeErr(model1:get(5).output:float() , model2:get(11).output:float()))
   end
   print('final error')
   print(relativeErr(model1.output:float() , model2.output:float()))

   -- check gradient error
   print('gradInput error 1st conv')
   print(relativeErr(spatial1.gradInput:float() , model2:get(1).gradInput:float()))
   print('gradInput error 1st Threshold')
   print(relativeErr(model1:get(2).gradInput:float() , model2:get(5).gradInput:float()))

   if twolayer then
      print('gradInput error 2nd conv')
      print(relativeErr(model1:get(3).gradInput:float() , model2:get(6).gradInput:float()))
   end

   print('weight error 1st conv')
   print(relativeErr(spatial1.weight:float() , model2:get(1).weightPreimage:select(5,1):float()))

   -- check weight error
   print('gradWeight error 1st conv')
   print(relativeErr(spatial1.gradWeight:float() , model2:get(1).gradWeightPreimage:select(5,1):float()))

   if twolayer then
      print('weight error 2nd conv')
      print(relativeErr(model1:get(4).weight:float() , model2:get(7).weightPreimage:select(5,1):float()))

      print('gradWeight error 2nd conv')
      print(relativeErr(model1:get(4).gradWeight:float() , model2:get(7).gradWeightPreimage:select(5,1):float()))
   end
end















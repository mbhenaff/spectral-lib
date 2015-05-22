-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'spectralnet'
require 'aux'

cmd = torch.CmdLine()
cmd:option('-dataset','reuters')
cmd:option('-model','gconv2','linear | gconv1 | gconv2 | fc2 | ... | fc5')
cmd:option('-optim','sgd')
cmd:option('-nhidden',64)
cmd:option('-k',5)
cmd:option('-rfreq',0,'reduction factor for freq bands')
cmd:option('-interp', 'spline','bilinear | spline | dyadic_spline | spatial')
cmd:option('-laplacian','gauss')
cmd:option('-poolsize',1)
cmd:option('-poolstride',1)
cmd:option('-poolneighbs',4)
cmd:option('-gpunum',1)
cmd:option('-printNorms',0)
cmd:option('-batchSize',128)
cmd:option('-learningRate',0.1)
cmd:option('-weightDecay',0)
cmd:option('-epochs',100)
cmd:option('-log',1)
cmd:option('-dropout',0)
cmd:option('-alpha',0.1)
cmd:option('-suffix','')
cmd:option('-normdata','feature')
cmd:option('-lambda',0)
cmd:option('-weightDecay',0)
cmd:option('-momentum',0)
cmd:option('-npts', 10, 'number of points to sample for commutation loss')
cmd:option('-interpScale',1)
opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(321)

assert(opt.optim == 'sgd' or opt.optim == 'adagrad')

if opt.log == 0 then 
   opt.log = false 
else
   opt.log = true
end

if opt.dropout == 0 then 
   opt.dropout = false 
else 
   opt.dropout = true
end

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/new/merck/'

opt.modelFile = 'dataset=' .. opt.dataset .. '-norm=' .. opt.normdata

opt.modelFile = opt.modelFile .. '-model=' .. opt.model .. '-batchSize=' .. opt.batchSize
if string.match(opt.model,'gconv') then 
   opt.modelFile = opt.modelFile
      .. '-interp=' .. opt.interp 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k
      .. '-laplacian=' .. opt.laplacian
      .. '-alpha=' .. opt.alpha
      .. '-interpScale=' .. opt.interpScale

   if opt.poolsize > 1 then
      opt.modelFile = opt.modelFile 
      .. '-poolsize-' .. opt.poolsize
      .. '-poolstride-' .. opt.poolstride
   end

elseif string.match(opt.model,'spatial') then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k  

   if opt.poolsize > 1 then
      opt.modelFile = opt.modelFile 
      .. '-poolsize-' .. opt.poolsize
      .. '-poolstride-' .. opt.poolstride
   end

elseif string.match(opt.model,'fc') or string.match(opt.model,'gc') then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
      .. '-weightDecay=' .. opt.weightDecay
end

if string.match(opt.model,'gc') then
   opt.modelFile = opt.modelFile ..'-lambda=' .. opt.lambda
      .. '-alpha=' .. opt.alpha
      .. '-npts=' .. opt.npts
end


if string.match(opt.model,'pool') then
   opt.modelFile = opt.modelFile 
      .. '-poolsize=' .. opt.poolsize
      .. '-poolstride=' .. opt.poolstride
      .. '-poolneighbs=' .. opt.poolneighbs
end
   

opt.modelFile = opt.modelFile .. '-optim=' .. opt.optim
opt.modelFile = opt.modelFile .. '-learningRate=' .. opt.learningRate

if opt.weightDecay ~= 0 then 
   opt.modelFile = opt.modelFile .. '-weightDecay=' .. opt.weightDecay
end

if opt.dropout then 
   opt.modelFile = opt.modelFile .. '-dropout'
end

if opt.suffix ~= '' then
   opt.modelFile = opt.modelFile .. '-' .. opt.suffix
end

opt.saveFile = opt.savePath .. opt.modelFile

print(opt.modelFile)
os.execute('mkdir -p ' .. opt.savePath)
logFileName = opt.savePath .. opt.modelFile .. '.log'

if opt.log then
   logFile = assert(io.open(logFileName,'w'))
   logFile:write(opt.modelFile .. '\n')
end

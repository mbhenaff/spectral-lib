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
cmd:option('-k',40)
cmd:option('-rfreq',0,'reduction factor for freq bands')
cmd:option('-interp', 'spline','bilinear | spline | dyadic_spline | spatial')
cmd:option('-laplacian','gauss')
cmd:option('-poolsize',1)
cmd:option('-poolstride',1)
cmd:option('-poolneighbs',4)
cmd:option('-gpunum',1)
cmd:option('-printNorms',0)
cmd:option('-batchSize',128)
cmd:option('-learningRate',0.05)
cmd:option('-weightDecay',0)
cmd:option('-epochs',500)
cmd:option('-log',1)
cmd:option('-dropout',0)
cmd:option('-alpha',0.1)
cmd:option('-suffix','')
cmd:option('-normdata','feature')
cmd:option('-lambda',0)
cmd:option('-weightDecay',0)
cmd:option('-momentum',0.9)
cmd:option('-npts', 10, 'number of points to sample for commutation loss')
cmd:option('-interpScale',1)
cmd:option('-testTime',0)
cmd:option('-stop',0)
opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(321)

assert(opt.optim == 'sgd' or opt.optim == 'adagrad')

opt.testTime = (opt.testTime == 1)
opt.log = (opt.log == 1)
opt.dropout = (opt.dropout == 1)
opt.stop = (opt.stop == 1)

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/new/merck/'

if opt.testTime then
   opt.savePath = opt.savePath .. '/testtime/'
end

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

elseif string.match(opt.model,'spatial') then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k  

elseif string.match(opt.model,'fc') or string.match(opt.model,'gc') then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
      .. '-weightDecay=' .. opt.weightDecay
end

if string.match(opt.model,'gc3') then
   opt.modelFile = opt.modelFile ..'-lambda=' .. opt.lambda
      .. '-alpha=' .. opt.alpha
      .. '-npts=' .. opt.npts
end


if string.match(opt.model,'pool') then
   opt.modelFile = opt.modelFile 
      .. '-poolsize=' .. opt.poolsize
      .. '-poolstride=' .. opt.poolstride
      --.. '-poolneighbs=' .. opt.poolneighbs
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

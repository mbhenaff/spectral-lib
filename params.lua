-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'cucomplex'
require 'Crop'
require 'Bias'
require 'Interp'
require 'SpectralConvolution'
require 'SpectralConvolutionImage'
require 'InterpImage'
require 'ComplexInterp'
require 'Real'
require 'GraphMaxPooling'
require 'loadData'
require 'FFTconv'
cufft = dofile('cufft/cufft.lua')


cmd = torch.CmdLine()
cmd:option('-dataset','mnist')
cmd:option('-model','gconv','mlp | gconv')
cmd:option('-optim','sgd')
cmd:option('-nhidden',64)
cmd:option('-k',5)
cmd:option('-rfreq',0,'reduction factor for freq bands')
cmd:option('-interp', 'bilinear','bilinear | spline | dyadic_spline | spatial')
cmd:option('-poolsize',4)
cmd:option('-poolstride',4)
cmd:option('-gpunum',1)
cmd:option('-printNorms',0)
cmd:option('-batchSize',32)
cmd:option('-learningRate',0.01)
cmd:option('-weightDecay',0)
cmd:option('-epochs',100)
cmd:option('-log',1)
cmd:option('-dropout',0)
cmd:option('-suffix','')
opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(321)

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

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/'

opt.modelFile = 'dataset=' .. opt.dataset .. '-model=' .. opt.model .. '-batchSize-' .. opt.batchSize
if string.match(opt.model,'gconv') or string.match(opt.model,'spectral') then 
   opt.modelFile = opt.modelFile
      .. '-interp=' .. opt.interp 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k
      .. '-poolsize-' .. opt.poolsize
      .. '-poolstride-' .. opt.poolstride
elseif string.match(opt.model,'spatial') then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k  
      .. '-poolsize-' .. opt.poolsize
      .. '-poolstride-' .. opt.poolstride
elseif opt.model == 'mlp' then
   opt.modelFile = opt.modelFile 
      .. '-nhidden=' .. opt.nhidden 
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

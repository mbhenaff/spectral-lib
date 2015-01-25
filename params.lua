-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'cucomplex'
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
cmd:option('-nhidden',32)
cmd:option('-k',25)
cmd:option('-rfreq',0,'reduction factor for freq bands')
cmd:option('-interp', 'bilinear','bilinear | spline | dyadic_spline | spatial')
cmd:option('-gpunum',1)
cmd:option('-batchSize',64)
cmd:option('-learningRate',0.1)
cmd:option('-weightDecay',0)
cmd:option('-epochs',20)
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
if opt.model == 'gconv' or opt.model == 'spectral' or opt.model == 'spectral2' then
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-model=' .. opt.model
      .. '-interp=' .. opt.interp 
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k
      .. '-learningRate=' .. opt.learningRate
else
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-model=' .. opt.model
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.k
      .. '-learningRate=' .. opt.learningRate
end

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

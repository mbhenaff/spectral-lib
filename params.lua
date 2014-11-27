-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'cucomplex'
require 'SpectralConvolution'
require 'Real'
require 'Crop'
require 'Bias'
require 'ComplexInterp'
require 'loadData'
require 'FFTconv'
cufft = dofile('cufft/cufft.lua')


cmd = torch.CmdLine()
cmd:option('-dataset','cifar')
cmd:option('-type','spectral','spatial | spectral')
cmd:option('-real','real','must be real or mod')
cmd:option('-nhidden',32)
cmd:option('-kH',5)
cmd:option('-kW',5)
cmd:option('-interp', 'spline')
cmd:option('-gpunum',1)
cmd:option('-batchSize',128)
cmd:option('-learningRate',0.1)
cmd:option('-weightDecay',0)
cmd:option('-ncrop',0, 'number of rows/cols to crop on each side after spectral conv')
cmd:option('-epochs',20)
opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(321)

if opt.real == 'real' then
   opt.realKernels = true
else 
   opt.realKernels = false
end

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/'
if opt.type == 'spectral' then
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-conv=' .. opt.type
      .. '-interp=' .. opt.interp 
      .. '-real=' .. opt.real
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.kH .. 'x' .. opt.kW 
      .. '-learningRate=' .. opt.learningRate
else
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-conv=' .. opt.type
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.kH .. 'x' .. opt.kW 
      .. '-learningRate=' .. opt.learningRate
end

opt.saveFile = opt.savePath .. opt.modelFile

print(opt.modelFile)
os.execute('mkdir -p ' .. opt.savePath)
logFileName = opt.savePath .. opt.modelFile .. '.log'

-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'spectralcuda'
require 'SpectralConvolutionImage'
require 'Real'
require 'Crop'
require 'Bias'
require 'ComplexInterp'
require 'loadData'
require 'FFTconv'
cufft = dofile('cufft/cufft.lua')


cmd = torch.CmdLine()
cmd:option('-dataset','cifar')
cmd:option('-conv','spectral','spatial | spectral')
cmd:option('-real','real','how to make output of spectral conv real (real | mod)')
cmd:option('-nhidden',32)
cmd:option('-kH',5)
cmd:option('-kW',5)
cmd:option('-interp', 'spline','bilinear | spline | dyadic_spline | spatial')
cmd:option('-gpunum',1)
cmd:option('-batchSize',128)
cmd:option('-learningRate',0.1)
cmd:option('-weightDecay',0)
cmd:option('-ncrop',0, 'number of rows/cols to crop on each side after spectral conv')
cmd:option('-epochs',20)
cmd:option('-log',1)
opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(321)

if opt.real == 'real' then
   opt.realKernels = true
else 
   opt.realKernels = false
end
if opt.log == 0 then
   opt.log = false
else
   opt.log = true
end

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/results/'
if opt.conv == 'spectral' then
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-conv=' .. opt.conv
      .. '-interp=' .. opt.interp 
      .. '-real=' .. opt.real
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.kH .. 'x' .. opt.kW 
      .. '-learningRate=' .. opt.learningRate
else
   opt.modelFile = 'dataset=' .. opt.dataset
      .. '-conv=' .. opt.conv
      .. '-nhidden=' .. opt.nhidden 
      .. '-k=' .. opt.kH .. 'x' .. opt.kW 
      .. '-learningRate=' .. opt.learningRate
end

opt.saveFile = opt.savePath .. opt.modelFile

print(opt.modelFile)
os.execute('mkdir -p ' .. opt.savePath)
logFileName = opt.savePath .. opt.modelFile .. '.log'

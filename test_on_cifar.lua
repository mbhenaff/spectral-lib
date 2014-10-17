require 'torch'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'cucomplex'
require 'SpectralConvolution'
cufft = dofile('cufft/cufft.lua')

torch.manualSeed(321)

cmd = torch.CmdLine()
cmd:option('-nhidden',16)
cmd:option('-sH',4)
cmd:option('-sw',4)
cmd:option('-gpunum',1)

opt = cmd:parse(arg or {})

cutorch.setDevice(opt.gpunum)
torch.setdefaulttensortype('torch.FloatTensor')

data,labels = loadData('cifar')
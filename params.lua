-- process input parameters 
require 'torch'
require 'image'
require 'gnuplot'
require 'optim'
require 'nn'
require 'cunn'
require 'cucomplex'
require 'SpectralConvolution'
require 'Modulus'
require 'Crop'
require 'Bias'
require 'ComplexInterp'
require 'loadData'
require 'FFTconv'
cufft = dofile('cufft/cufft.lua')


cmd = torch.CmdLine()
cmd:option('-type','spectral','spatial | spectral')
cmd:option('-nhidden',96)
cmd:option('-kH',4)
cmd:option('-kW',4)
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

opt.savePath = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/'
opt.modelFile = opt.type .. '-' .. opt.interp 
                         .. '-nhidden=' .. opt.nhidden 
                         .. '-k=' .. opt.kH .. 'x' .. opt.kW 
                         .. '-learningRate=' .. opt.learningRate

opt.saveFile = opt.savePath .. opt.modelFile

print(opt.modelFile)
os.execute('mkdir -p ' .. opt.savePath)
logFileName = opt.savePath .. opt.modelFile .. '.log'

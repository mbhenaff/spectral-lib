dofile('params.lua')

iH = 28
iW = 28
nhidden = 32

L = torch.load('mresgraph/mnist_laplacian_spatialsim_poolsize_9_stride_4_neighbs_9.th')
--L = torch.load('mresgraph/mnist_laplacian_spatialsim_poolsize_4_stride_4_neighbs_4.th')
--L = torch.load('mresgraph/mnist_laplacian_poolsize_4_stride_4.th')
V1 = L.V[1]:float()
V2 = L.V[2]:float()

inputs = torch.randn(opt.batchSize, 1, iH, iW):cuda()
model1 = nn.SpectralConvolutionImage(opt.batchSize, 1, opt.nhidden, iH, iW, opt.k, opt.k, opt.interp, 'real'):cuda()
model1:reset()
out1 = model1:updateOutput(inputs)
out1 = out1:select(5,1):clone()
out1:resize(opt.batchSize,iH*iW*opt.nhidden)
norms1=out1:norm(2,2):float():squeeze()

dim = iH*iW
inputs = inputs:resize(opt.batchSize, 1, dim)
model2 = nn.SpectralConvolution(opt.batchSize, 1, opt.nhidden, dim, opt.k*opt.k, V1):cuda()
model2:reset()
out2 = model2:forward(inputs)
out2:resize(opt.batchSize,iH*iW*opt.nhidden)
norms2=out2:norm(2,2):float():squeeze()


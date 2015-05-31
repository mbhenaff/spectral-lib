require 'datasource'
datasource = Datasource(opt.dataset, opt.normdata, opt.testTime)
datasource:type('torch.CudaTensor')
dim = datasource.dim
nChannels = datasource.nChannels
nclasses = datasource.nClasses
trsize = datasource.nSamples['train']
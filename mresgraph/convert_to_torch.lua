matio = require 'matio'
dataset = 'cifar'
poolsize = 4
stride = 4
neighbs = 4
fname = dataset .. '_laplacian_poolsize_' .. poolsize .. '_stride_' .. stride .. '_neighbs_' .. neighbs
x = matio.load(fname .. '.mat')
torch.save(fname .. '.th',{V=x.V, pools=x.pools})
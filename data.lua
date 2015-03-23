-- load dataset
if opt.dataset == 'reuters' then
   nclasses = 50
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
   dim = trdata:size(2)
   nChannels = 1
elseif opt.dataset == 'mnist' then
   nclasses = 10
   trdata,trlabels = loadData(opt.dataset,'train',true)
   tedata,telabels = loadData(opt.dataset,'test',true)
   dim = trdata:size(3)
   nChannels = 1
elseif opt.dataset == 'cifar' then
   nclasses = 10
   trdata,trlabels = loadData(opt.dataset,'train',true)
   tedata,telabels = loadData(opt.dataset,'test',true)
   dim = trdata:size(3)
   nChannels = trdata:size(2)
elseif opt.dataset == 'timit' then
   nclasses = 185
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'dev')
   dim = trdata:size(2)
end

if trdata:nDimension() == 4 then 
   trdata = trdata:resize(trdata:size(1),trdata:size(2), trdata:size(3)*trdata:size(4))
   tedata = tedata:resize(tedata:size(1),tedata:size(2), tedata:size(3)*tedata:size(4))
end

nsamples = trdata:size(1)

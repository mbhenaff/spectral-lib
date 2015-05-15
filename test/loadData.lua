-- script to load MNIST or CIFAR

require 'image'
--require 'utils'
require 'memoryMap'

function loadData(dataset,split,resize,normalize)
   local resize = resize or false
   local f, data, labels
   if dataset == 'cifar' then
      local cifar = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/cifar_10_norm.th')
      --local cifar = torch.load('/misc/vlgscratch3/LecunGroup/CIFAR.t7')
      f = {}
      if split == 'train' then 
         --f = cifar.tr
         f.data = cifar.trdata
         f.labels = cifar.trlabels
      elseif split == 'test' then
         --f = cifar.test_data
         f.data = cifar.tedata
         f.labels = cifar.telabels
      end
      if resize then
         data:resize(data:size(1), data:size(2), data:size(3)*data:size(4))
      end
   elseif dataset == 'mnist' then
      if split == 'train' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/train_28x28.th7nn')
         --f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/train_32x32.th7n')
      elseif split == 'test' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/test_28x28.th7nn')
         --f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/test_32x32.th7n')
      else
         error('set should be train or test')
      end
      labels = f.labels
      data = f.data
      --data:div(255)
      if resize then
         data:resize(data:size(1), data:size(2)*data:size(3)*data:size(4))
      end
   elseif dataset == 'reuters' then
      if split == 'train' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/train.th')
      elseif split == 'test' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/test.th')
      else
         error('set should be train or test')
      end

      local means
      local std
      if not paths.dirp('/scratch/mbhenaff/reuters') then
         paths.mkdir('/scratch/mbhenaff/reuters')
      end
      
      if paths.filep('/scratch/mbhenaff/reuters/means.th') then
         means = torch.load('/scratch/mbhenaff/reuters/means.th')
      else
         means = torch.mean(f.data,1)
         torch.save('/scratch/mbhenaff/reuters/means.th',means)
      end
      if paths.filep('/scratch/mbhenaff/reuters/std.th') then
         std = torch.load('/scratch/mbhenaff/reuters/std.th')
      else
         std = torch.std(f.data,1)
         torch.save('/scratch/mbhenaff/reuters/std.th',std)         
      end
      if normalize then
         f.data:add(-1,means:expandAs(f.data))
         f.data:cdiv(std:expandAs(f.data))
      end

      labels = f.labels
      data = f.data

   elseif dataset == 'imagenet' then
      f = {}
      if split == 'test' then 
         f.labels = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/telabels.th')
         f.data = torch.loadMemoryFile('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/tedata.th','torch.FloatTensor')
      elseif split == 'train1' then
         f.labels = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trlabels1.th')
         f.data = torch.loadMemoryFile('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trdata1.th','torch.FloatTensor')
      elseif split == 'train2' then
         f.labels = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trlabels2.th')
         f.data = torch.loadMemoryFile('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trdata2.th','torch.FloatTensor')
      elseif split == 'train3' then
         f.labels = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trlabels3.th')
         f.data = torch.loadMemoryFile('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trdata3.th','torch.FloatTensor')
      elseif split == 'train4' then
         f.labels = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trlabels4.th')
         f.data = torch.loadMemoryFile('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/trdata4.th','torch.FloatTensor')
      else
         error('unrecognized split')
      end
      if resize then
         f.data:resize(100000,3,128,128)
      end
   elseif dataset == 'timit' then
      if paths.filep('/scratch/timit/' .. split .. '/data_winsize_15.th') then
         f = torch.load('/scratch/timit/' .. split .. '/data_winsize_15.th')
      else
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/torch_datasets/timit/' .. split .. '/data_winsize_15.th')
      end
      print(f)
      local nSamples = f.data:size(1)
      local nFrames = f.data:size(2)
      local nBands = f.data:size(3)
      f.data:resize(nSamples,nFrames*nBands)
      f.labels = f.labels + 1
      
      local means
      local std
      if not paths.dirp('/scratch/mbhenaff/timit') then
         paths.mkdir('/scratch/mbhenaff/timit')
      end
      
      if paths.filep('/scratch/mbhenaff/timit/means.th') then
         means = torch.load('/scratch/mbhenaff/timit/means.th')
      else
         means = torch.mean(f.data,1)
         torch.save('/scratch/mbhenaff/timit/means.th',means)
      end
      if paths.filep('/scratch/mbhenaff/timit/std.th') then
         std = torch.load('/scratch/mbhenaff/timit/std.th')
      else
         std = torch.std(f.data,1)
         torch.save('/scratch/mbhenaff/timit/std.th',std)         
      end

      f.data:add(-1,means:expandAs(f.data))
      f.data:cdiv(std:expandAs(f.data))
      f.data = f.data:resize(nSamples, nFrames, nBands)
   end
   local labels = f.labels
   local data = f.data

   return data:float(),labels:float()
end


-- show some images (note, we're assuming the data is [nSamples x n*n])
function showImages(data,nrows)
   local nSamples = data:size(1)
   local N = math.sqrt(data:size(2))
   imgs = data:resize(nSamples,N,N)
   image.display{image=imgs,nrows = nrows}
end


function reorder(data, indx)
   local d = data:size(2)
   if indx:nElement() ~= d then
      error('indx must equal dim')
   end
   local data2 = torch.Tensor(data:size())
   for i = 1,d do 
      data2[{{},i}]:copy(data[{{},indx[i]}])
   end
   return data2
end

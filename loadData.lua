-- script to load MNIST or CIFAR

require 'image'
require 'utils'
require 'memoryMap'

function loadData(dataset,split, indx)
   local f, data, labels
   if dataset == 'cifar' then
      local cifar = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/cifar_10_norm.t7')
      if split == 'train' then 
         f = cifar.train
      elseif split == 'test' then
         f = cifar.test
      end
      labels = f.labels
      data = f.data
      data:resize(data:size(1), data:size(2)*data:size(3)*data:size(4))
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
      --data:resize(data:size(1), data:size(2)*data:size(3)*data:size(4))
   elseif dataset == 'reuters' then
      if split == 'train' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/train.th')
      elseif split == 'test' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/test.th')
      else
         error('set should be train or test')
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
      f.data:resize(100000,3,128,128)
      if false then
      local dict = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/imagenet_small/labels_dict.th')
      for i = 1,f.labels:size(1) do 
         f.labels[i] = dict[f.labels[i]]
      end
      end
      --f.labels=f.labels[{{1,1000}}]
      --f.data=f.data[{{1,1000}}]
   end
   local labels = f.labels
   local data = f.data

   if false then
      for i = 1,data:size(1) do 
         if i % 1000 then 
            print(i)
         end
         data[i]:add(-data[i]:mean())
         data[i]:mul(math.max(1/data[i]:std(),0.00001))
      end
   end

   if indx ~= nil then
      print('reordering')
      data = reorder(data, indx)
   end
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

-- script to load MNIST or CIFAR

require 'image'
require 'utils'

function loadData(dataset,split)
   local f
   if dataset == 'cifar' then
      local cifar = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/cifar_10_norm.t7')
      if split == 'train' then 
         f = cifar.train
      elseif split == 'test' then
         f = cifar.test
      end
   elseif dataset == 'mnist' then
      if split == 'train' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/train_28x28.th7nn')
      elseif split == 'test' then
         f = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/test_28x28.th7nn')
      else
         error('set should be train or test')
      end
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

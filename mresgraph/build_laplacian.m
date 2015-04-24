addpath(genpath('../../../vlfeat-0.9.16'));

dataset = 'reuters';

fprintf('loading data...');
if strcmp(dataset, 'timit')
  x=load('/misc/vlgscratch3/LecunGroup/mbhenaff/timit/fbanks/train/data_winsize_15.mat');
  data = x.data;
elseif strcmp(dataset, 'cifar')
  x=load('/misc/vlgscratch3/LecunGroup/mbhenaff/cifar-10-batches-t7/trdata.mat');
  data = x.trdata;
  data = permute(data,[4 3 2 1]);
  data = data(:,1,:,:);
  data = reshape(data, [50000 1*32*32]);
elseif strcmp(dataset,'mnist')
  x=load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/trdata.mat');
  data = reshape(x.x,[28 28 60000]);
  data = permute(data,[3 2 1]);
  data = reshape(data,[60000 28*28]);
elseif strcmp(dataset, 'reuters')
  data=load('/misc/vlgscratch3/LecunGroup/mbhenaff/reuters_50/train.mat','data');
  data = data.data;
elseif strcmp(dataset, 'mnist_spatialsim')
  data = zeros(2,784);
  for i=1:28
    for j=1:28
      data(1,(i-1)*28+j) = i;
      data(2,(i-1)*28+j) = j;
    end
  end
elseif strcmp(dataset, 'cifar_spatialsim')
  data = zeros(2,32*32);
  for i=1:32
    for j=1:32
      data(1,(i-1)*32+j) = i;
      data(2,(i-1)*32+j) = j;
    end
  end
end
fprintf('done\n')


% normalize the data
mu = mean(data,1);
data = data - repmat(mu,size(data,1),1);
sigmas = std(data,0,1);
data = data./repmat(sigmas,size(data,1),1);





% data is now [nsamples x nfeatures]
poolsize = 8;
stride = 4;
neighbs = 200;
[V,pools,W1,W2] = build_graph_laplacian_space(data,poolsize, stride, neighbs);
save(['/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/' dataset '_laplacian_poolsize_' num2str(poolsize) '_stride_' num2str(stride) '_neighbs_' num2str(neighbs) '.mat'],'V','pools','W1','W2');

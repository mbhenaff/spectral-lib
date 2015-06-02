function data=loadData(dataset)
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
  data = log(data+1);
  norms = sqrt(sum(data.^2,2));
  data = data ./ repmat(norms,[1 size(data,2)]);
%  fprintf('normalizing...');
%  mu = mean(data,1);
%  data = data - repmat(mu,size(data,1),1);
 % sigmas = std(data,0,1);
%  data = data./repmat(sigmas,size(data,1),1);
  fprintf('done\n')
elseif strcmp(dataset,'reutersfc')
data=load('reuters-reutersfc.mat')
data=data.W1';

elseif strcmp(dataset,'merck3fc')
 data=load('dnn4.mat');
 data=data.W1';
elseif strcmp(dataset,'merck3grad')
 data=load('merck3-dnn4.mat');
 data=data.G';
elseif strcmp(dataset,'merck3fc2')
 data=load('dnn4.mat');
 data=(data.W1*data.W2)';
elseif strcmp(dataset,'merck3fc3')
 data=load('dnn4.mat');
 data=(data.W1*data.W2*data.W3)';
elseif strcmp(dataset,'merck3fc4')
 data=load('dnn4.mat');
 data=(data.W1*data.W2*data.W3*data.W4)';

elseif ~isempty(findstr(dataset,'merck'))
  num = regexp(dataset,'\d+','match')
  data=load(['/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/' dataset '_train.mat'],'data');
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
end
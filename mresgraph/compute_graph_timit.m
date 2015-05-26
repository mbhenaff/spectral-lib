addpath(genpath('releaseCode'));
alpha = 0.0001;
loaded = true;
if ~loaded
  load('/misc/vlgscratch3/LecunGroup/mbhenaff/timit/fbanks/train/data_winsize_15.mat');
  data = reshape(data,size(data,1)*15,120);
  shuffle = randperm(size(data,1));
  data = data(shuffle,:);
end
fprintf('Data loaded\n')


n = 10000;
nFeatures = 120;

W = zeros(nFeatures, nFeatures);
params = {}
params.sigx = -1;
params.sigy = -1;
params.q = 4;

for i = 1:nFeatures
  for j = 1:nFeatures
    disp([i j]);
    [thresh, testStat] = GreGyoL1Test(data(1:n,i),data(1:n,j),alpha,params);
    if(testStat > thresh)
      W(i,j)=1;
    end
  end
end

    

    










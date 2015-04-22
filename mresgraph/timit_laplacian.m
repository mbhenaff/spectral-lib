%this script generates several laplacians for the TIMIT dataset

if 0
clear all
close all

load('/misc/vlgscratch3/LecunGroup/mbhenaff/timit/fbanks/train/data_winsize_15.mat');

data=data';
data=reshape(data,120,numel(data)/120);

fprintf('data ready \n')
end
%center and diagonally normalize data 
mu=mean(data,2);
data = data - repmat(mu,1,size(data,2));
sigmas = std(data,0,2);
data = data./repmat(sigmas,1,size(data,2));

%%1 Use the euclidean Kernel (aka PCA)
G = data*data';
[V0,D0]=eig(G);

%%2 Gaussian Kernel
K0 = kernelization(data);
%adjust sparsity level 
alpha=0.1;
[K0s, Is]=sort(K0,2,'ascend');
loc=round(alpha*size(K0s,2));
sigma = sqrt( mean(K0s(:,loc)));
K1=exp(-K0/sigma^2);
D = diag(sum(K1).^(-1/2));
L = eye(size(K1,1)) - D * K1 * D;
[V1,ev]=eig(L);



%%3 Random Kernel

LW=randn(size(L));
[V2,~,~]=svd(LW);


%construct the neighbordhoods
NN=zeros(size(L));
for i=1:size(K0s,1)
ind = find(K0(i,:) < sigma^2);
NN(i,ind)=1;
end


save('/misc/vlgscratch3/LecunGroup/bruna/timit_laplacians.mat','V0','V1','V2','NN');



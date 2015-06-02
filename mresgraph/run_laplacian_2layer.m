%this script generates the GFT matrices and pools for 2 layers
dataset = 'reutersfc';
kernel = 'gauss';
poolsize = [8 8];
poolstride = [4 4];
alpha=0.01;
r = 0.25;

if 1
%  clear all
  close all
  data = loadData(dataset);
  data=data';

  if strcmp(dataset,'timit')
    data=reshape(data,120,numel(data)/120);
  end
  fprintf('data ready \n')

if strcmp(kernel,'gauss')
%%2 Gaussian Kernel
fprintf('computing kernel...');
if 1
[U,S,V]=svd(data,0);
nComp = r*min(size(data));
data=U(:,1:nComp)*S(1:nComp,1:nComp)*V(:,1:nComp)';
end
K0 = kernelization(data);
%adjust sparsity level 
[K0s, Is]=sort(K0,2,'ascend');
loc=round(alpha*size(K0s,2));
sigma = sqrt( mean(K0s(:,loc)));
K1=exp(-K0/sigma^2);
fprintf('done\n');
% compute laplacian
D = diag(sum(K1).^(-1/2));
L = eye(size(K1,1)) - D * K1 * D;
L = (L+L')/2;
[V1,ev]=eig(L);

%construct the neighbordhoods
NN=zeros(size(L));
for i=1:size(K0s,1)
  ind = find(K0(i,:) < sigma^2);
  NN(i,ind)=1;
end

      elseif strcmp(kernel,'gausslocal')
      %%2 Gaussian Kernel
      fprintf('computing kernel...');
      K0 = kernelization(data);
                                %adjust sparsity level 
      [K0s, Is]=sort(K0,2,'ascend');
      loc=round(alpha*size(K0s,2));
      sigma = sqrt( mean(K0s(:,loc)));
      sigmas = sqrt(abs(K0(:,loc)));
      K1 = exp(-K0 ./ (sigmas*sigmas'));
      %K1=exp(-K0/sigma^2);
      fprintf('done\n');
                                % compute laplacian
      D = diag(max(0.000001,sum(K1)).^(-1/2));
      L = eye(size(K1,1)) - D * K1 * D;
      L = (L+L')/2;
      [V1,ev]=eig(L);
                                %construct the neighbordhoods
      NN=zeros(size(L));
      for i=1:size(K0s,1)
        ind = find(K0(i,:) < sigma^2);
        NN(i,ind)=1;
      end
        
end


end


% Get pools 
[V,pools] = ms_spectral_clustering(K1,K1,2,1./poolstride,poolsize);
path = '/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/'; 
save([path 'alpha_' num2str(alpha) '/' dataset '_laplacian_' kernel '_pool1_' num2str(poolsize(1)) '_stride1_' num2str(poolstride(1)) '_pool2_' num2str(poolsize(2)) '_stride2_' num2str(poolstride(2)) 'rank_' num2str(r) '.mat'],'V','NN','pools');



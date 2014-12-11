function [ker,ndata]=gkernel(data,options)

NN=getoptions(options,'gNN',0.3);
NNA=getoptions(options,'NNA',16);

ker=kernelization(data);

%sigma should be some sort of a rank estimate
L=size(ker,1);
ks=sort(ker(:),'ascend');
ks=ks(L+1:end);
posi = max(round(L*L*NN),round(L*NNA));
sigma = sqrt(ks(posi));

rho=getoptions(options,'maxcoherence',0.999);

%neighbs = sum((ker < 2*(1-rho)),2);
neighbs = sum((ker <= sigma^2),2);
ndata = data ./ (neighbs*ones(1,size(data,2)));

fprintf('sigma is estimated at %f \n',sigma)

ker=exp(-ker/(2*sigma^2));
%ndata=data;





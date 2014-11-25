function out=spatial_kern(K,N)
%this function simply maps compact support kernels into the fourier domain

%N: number of parameters to learn (downsampled grid)
%M: size of input 

%Q=2;

out=zeros(K^2,N^2);

mask=zeros(N);
for n1=1:K
for n2=1:K
mask=0*mask;
mask(n1-round(K/2),n2-round(K/2))=1;
tmp=fft2(mask);
out(n1+K*(n2-1),:)=tmp(:);
end
end



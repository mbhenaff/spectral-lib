function out=spline_interp_kern_dyadic(N,M)
%this function constructs spline interpoaltion 
%using dyadic grid and taking care of borders

%N: number of parameters to learn (downsampled grid)
%M: size of input 

%Q=2;

out=zeros(N,M);

%x=linspace(0,1,N);
if mod(N,2)==0

x(1)=0;
x(2:N/2+1)=2.^[-N/2:-1];
x(N/2+2:N)=1-fliplr(x(2:N/2));
x(N+1)=1;

else

l=(N-1)/2;
beta=log2(1/3);
x(1)=0;
x(2:l+1)=2.^([-N/2+1:0]+beta);
x(l+2:2*l+2)=1-fliplr(x(1:l+1));

end

y=linspace(0,1,M);

%we also impose 0 derivative in the borders
aux=zeros(1,N+3);
for n=1:N
aux=0*aux;
aux(n+1)=1;
if n==1
aux(N+2)=1;
end
out(n,:)=spline(x,aux,y);
end



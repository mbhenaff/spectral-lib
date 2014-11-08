function out=spline_interp_kern(N, M)
%this function constructs a linear interpolation kernel using cubic splines

out=zeros(N,M);

x=linspace(0,1,N);
y=linspace(0,1,M);
aux=zeros(1,N);
for n=1:N
aux=0*aux;
aux(n)=1;
out(n,:)=spline(x,aux,y);
end



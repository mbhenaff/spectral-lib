function ker=kernelization(data)

[L,N]=size(data);

norms=sum(data.^2,2)*ones(1,L);
ker=norms+norms'-2*data*(data');


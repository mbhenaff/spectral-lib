function [ker,kern]=kernelizationbis(data,databis)

    [L,N]=size(data);
    [M,N]=size(databis);

    norms=sum(data.^2,2)*ones(1,M);
    normsbis=sum(databis.^2,2)*ones(1,L);
    ker=norms+normsbis'-2*data*(databis');
end


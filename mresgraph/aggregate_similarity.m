function wout=aggregate_similarity(win, poolsize, stride)
%this function aggregates similarity within pools. it assumes features have been reordered along a 1d topology

N=size(win,1);
kern=ones(poolsize);
wout=conv2(full(win),kern,'valid');
wout=wout(1:stride:end,1:stride:end);




addpath(genpath('../../../vlfeat-0.9.16'));

dataset = 'reuters';

data = loadData(dataset)

% data is now [nsamples x nfeatures]
poolsize = 8;
stride = 4;
neighbs = 200;
[V,pools,W1,W2] = build_graph_laplacian_space(data,poolsize, stride, neighbs);
W1 = full(W1);
W2 = full(W2);
save(['/misc/vlgscratch3/LecunGroup/mbhenaff/spectralnet/mresgraph/' dataset '_laplacian_poolsize_' num2str(poolsize) '_stride_' num2str(stride) '_neighbs_' num2str(neighbs) '.mat'],'V','pools','W1','W2');

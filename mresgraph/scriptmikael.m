% load data
%addpath(genpath('../../../vlfeat-0.9.16'));
x=load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/trdata.mat');
data = reshape(x.x,[28 28 60000]);
data = permute(data,[3 2 1]);
data = reshape(data,[60000 28*28]);

% data is now 60000 x 784
data = double(data);

poolsize = 9;
stride = 4;
[V,pools] = build_graph_laplacian_space(data,poolsize, stride);



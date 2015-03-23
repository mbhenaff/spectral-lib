function [V,perm,w,wpool] = build_graph_laplacian(T,poolsize,stride)

vl_setup


%% build a spatial hierarchical clustering
tree=vl_kdtreebuild(T);
j1 = 16;
j2 = 64;
ep = [0.05 0.1 0.05 0.1];

[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j1,'MAXCOMPARISONS',250) ;

fprintf('here')
opts.kNN=j1;opts.alpha=1;opts.kNNdelta=j1;w=fgf(T',opts,nnid');
fprintf(' there \n')
%[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j2,'MAXCOMPARISONS',500) ;
%opts.kNN=j2;opts.alpha=1;opts.kNNdelta=j2;wfat=fgf(T',opts,nnid');

%[smallblocks fatblocks wout]=ms_gc_bu_bis(w,wfat,ep,length(ep));
%
%for i=1:size(fatblocks,2)
%    clear blocks;
%    blocks = fatblocks{i};
%    save(sprintf(fullfile(matdir,'mnistsphereblocks_fat_l%d.mat'),i),'blocks');
%    clear blocks;
%    blocks = smallblocks{i};
%    save(sprintf(fullfile(matdir,'mnistsphereblocks_thin_l%d.mat'),i),'blocks');
%end
%

%%build the spectrum

D = diag(sum(w).^(-1/2));
L = eye(size(w,1)) - D * w * D;
[ee,ev]=eig(L);
W0=ee;

fprintf('here 2\n');
%compute reordering
options.null=0;
perm = 1:size(w,1);
%[perm, invperm] = haarpartition_graph(w,options);
V{1} = W0(perm,:);
w=w(perm,perm);

fprintf('here 3\n');

%compute new similarity
wpool = aggregate_similarity(w, poolsize, stride);

D = diag(sum(wpool).^(-1/2));
L = eye(size(wpool,1)) - D * wpool * D;
[ee,ev]=eig(L);
V{2}=ee;
fprintf('here4\n');









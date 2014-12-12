function W0 = build_graph_laplacian(T)

vl_setup


%% build a spatial hierarchical clustering
tree=vl_kdtreebuild(T);
j1 = 16;
j2 = 64;
ep = [0.05 0.1 0.05 0.1];

[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j1,'MAXCOMPARISONS',500) ;
opts.kNN=j1;opts.alpha=1;opts.kNNdelta=j1;w=fgf(T',opts,nnid');
[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j2,'MAXCOMPARISONS',500) ;
opts.kNN=j2;opts.alpha=1;opts.kNNdelta=j2;wfat=fgf(T',opts,nnid');

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
W1=W0';



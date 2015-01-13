function [V,pools] = build_graph_laplacian_space(T,poolsize,stride)
%T contains the data, in the format [features, examples]. 
%we construct pools with a hierarchical specrtal clustering. 
%poolsize: size of each pool
%stride: ratio of points before and after pooling

vl_setup

%% build a spatial hierarchical clustering
tree=vl_kdtreebuild(T);
frac = stride / poolsize;
j1 = round(poolsize * frac);
j2 = poolsize;
depth = 2;
psize = poolsize * ones(1,depth);

[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j1,'MAXCOMPARISONS',250) ;
opts.kNN=j1;opts.alpha=1;opts.kNNdelta=j1;w=fgf(T',opts,nnid');

[nnid, ndist] = vl_kdtreequery(tree,T,T, 'NUMNEIGHBORS',j2,'MAXCOMPARISONS',500) ;
opts.kNN=j2;opts.alpha=1;opts.kNNdelta=j2;wfat=fgf(T',opts,nnid');

%this function will do spectral clustering on w, then for each cluster, we pick the closest 
%exemplar to the centroid, and define the pool as its fat nearest neighbors. Once we have the pools, 
%we recompute similarities by adding them.
[V, pools] = ms_spectral_clustering(w,wfat,depth,1/stride, psize); 









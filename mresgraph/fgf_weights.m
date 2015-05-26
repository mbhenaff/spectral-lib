function [w NNIdxs NNDist]=fgf_weights(A,opts)
%[w NNIdxs NNDist]=fgf(A,opts,NNIdxs)
%implementation of self-tuning 
%weight matrix construction, as in Perona and Zelnik-Manor 2004
%using the tstools nearest neighbor searcher
%opts.quiet=1 means progress is not displayed; defaults to 0.
%opts.kNN is the number of neighbors
%opts.alpha is a universal scaling factor
%opts.kNNdelta specifies which nearest neighbor determines the local scale
%w is forced to be symmetric by averaging with its adjoint.
%note opts.kNNdelta=0 corresponds to standard (non-self-tuning)
%construction with variance parameter opts.alpha
%if NNIdxs (Nearest Neighbor Indices) given, uses these instead of running
%the tstools neighbor search. 

[N NN]=size(A);

if isfield(opts,'quiet')
    if opts.quiet==1
        quiet=1;
    else
        quiet=0;
    end
else
    quiet=0;
end

[Wsort,Wind]=sort(sqrt(A),2,'ascend');

NNIdxs = Wind(:,2:1+opts.kNN);
NNDist = Wsort(:,2:1+opts.kNN);

       
if opts.kNNdelta>0
    if opts.kNNdelta>opts.kNN
        sigma=ones(size(NNDist,1),1)';
    else
        sigma=NNDist(:,opts.kNNdelta)';
    end
else
    sigma=ones(size(NNDist,1),1)';
%    sigma=inf(size(NNDist,1),1)';
end

idx=1;
idxi=[];
idxj=[];
idxi(opts.kNN*N) = int16(0);
idxj(opts.kNN*N) = int16(0);
entries=zeros(1,opts.kNN*N);
for lk = 1:N
    idxi(idx:idx+opts.kNN-1)=lk;
    idxj(idx:idx+opts.kNN-1)=NNIdxs(lk,:);
    entries(idx:idx+opts.kNN-1)=exp(-(NNDist(lk,:).^2)./(eps+opts.alpha*sigma(lk)*sigma(NNIdxs(lk,:))));
    idx=idx+opts.kNN;
end;
w=sparse(idxi,idxj,entries,N,N);
w=(w+w')/2;


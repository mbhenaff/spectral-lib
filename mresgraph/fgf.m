function [w NNIdxs NNDist]=fgf(A,opts,NNIdxs,NNDist)
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

[N dim]=size(A);

if isfield(opts,'quiet')
    if opts.quiet==1
        quiet=1;
    else
        quiet=0;
    end
else
    quiet=0;
end


if isfield(opts,'NNcoordinates')
    if length(opts.NNcoordinates)>0
        NNcoordinates=opts.NNcoordinates;
        coarse=1;
    else
        coarse=0;
        NNcoordinates=[1:dim];
    end
else
    coarse=0;
    NNcoordinates=[1:dim];
end


if nargin==2
    atria=nn_prepare(A(:,NNcoordinates));
    NNIdxs=zeros(N,opts.kNN); 
    NNDist=zeros(N,opts.kNN);
    ccount=1;
    for zz=1:ceil(N/2000)
        if ccount+2000>N;
        cdiff=N-ccount;
        ccount=N;
        else
            cdiff=2000;
            ccount=ccount+2000;
        end
        if quiet==0
            zz
        end
        [NNIdxs(ccount-cdiff:ccount,:) NNDist(ccount-cdiff:ccount,:)]=nn_search(A(:,NNcoordinates),atria,A(ccount-cdiff:ccount,NNcoordinates),opts.kNN);
    end                               

    if coarse==1
        for k=1:N
            for j=1:opts.kNN
                NNDist(k,j)=norm(A(k,:)-A(NNIdxs(k,j),:));
            end
        end
    end
elseif nargin<4
    NNDist=zeros(N,opts.kNN);
    if issparse(A)
       [r c v]=find(A);
       [b idx]=mvi(r);
        for k=1:N 
            if quiet==0
                if mod(k,2000)==1
                    k
                end
            end
            for j=1:opts.kNN
                [s s1 s2]=intersect(c(idx{k}),c(idx{NNIdxs(k,j)}));
                sv1=v(idx{k});
                sv2=v(idx{NNIdxs(k,j)});
                overlapsum=sum(((sv1(s1))-sv2(s2)).^2);
                sv1(s1)=0;
                sv2(s2)=0;
                NNDist(k,j)=sqrt(overlapsum+sum(sv1.^2)+sum(sv2.^2));
            end
        end
    else
        
        for k=1:N
           for j=1:opts.kNN
               NNDist(k,j)=norm(A(k,:)-A(NNIdxs(k,j),:));
           end
        end
   end
    
end



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



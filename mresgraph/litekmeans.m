function [outlabel,outm] = litekmeans(X, k, constrained, maxsize)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
% Modified by Joan Bruna to have constant cluster sizes

n = size(X,2);
last = 0;

minener = 1e+20;
outiters=5;
maxiters=1000;

if nargin < 4
maxsize = n/k;
end


for j=1:outiters
    %printf(2, 'Iter %d / %d\n', j, outiters);
    s = RandStream('mt19937ar','Seed',j);
    aux=randperm(s, n);
    m = X(:,aux(1:k));
	ww=kernelizationbis(X',m');
	if ~constrained
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
	else
    [label] = constrained_assignment(X, m,maxsize,ww);
	end

    iters=0;


    while any(label ~= last) & iters < maxiters
        [u,~,label] = unique(label);   % remove empty clusters
        k = length(u);
        E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
        m = X*full(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
	ww=kernelizationbis(X',m');
		if ~constrained
        last = label';
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
		else
        last = label;
        [label] = constrained_assignment(X, m,maxsize,ww);
		end
		ener=0;
		for n=1:size(ww,1)
		ener=ener+ww(n,label(n));
		end


        iters = iters +1 ;                
    end
    
    [~,~,label] = unique(label);            
    
    if ener < minener
        outlabel = label;
        outm = m;
        minener = ener;
    end            
	fprintf('done iter %d ener %f \n', j, ener)

end


end



function [out]=constrained_assignment(X, C, K, w)
%we assign samples to the nearest centers, but with the constraint that each center receives K samples

[N,M]=size(w); %N number of samples, M number of centers

maxvalue = max(w(:))+1;

[ds,I]=sort(w,2,'ascend');
%[ds2,I2]=sort(w,1,'ascend');

out=I(:,1);
for m=1:M
    taille(m)=length(find(out==m));
end
[hmany,nextclust]=max(taille);

visited=zeros(1,M);


go=(hmany > K);
choices=ones(N,1);

while go
    %fprintf('%d %d \n', nextclust, hmany)
    aux=find(out==nextclust);

    for l=1:length(aux)
        slice(l) = ds(aux(l),choices(aux(l))+1)-ds(aux(l),choices(aux(l)));
    end
    [~,tempo]=sort(slice,'descend');
    clear slice;
    %slice=w(aux,nextclust);
    %[~,tempo]=sort(slice,'ascend');
    
    saved=aux(tempo(1:K));
    out(saved)=nextclust;

    visited(nextclust)=1;
    for k=K+1:length(tempo)
       i=2;
       while visited(I(aux(tempo(k)),i)) 
          i=i+1;
       end
       out(aux(tempo(k)))=I(aux(tempo(k)),i);
       choices(aux(tempo(k)))=i;
    end
    for m=1:M
        taille(m)=length(find(out==m));
    end
    [hmany,nextclust]=max(taille);
    go=(hmany > K);
end

end




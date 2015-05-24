function W = lowranksimilarity(data)
%we attempt to factorize pairwise densities with marginals. 

%%
[L,N] = size(data);

nbins = 256;
code=zeros(L,N);

%%obtain bins on each marginal
for l=1:L
	[h{l}, bins{l}]=hist(data(l,:),nbins);
	ref = repmat(bins{l}',1,N);
	code(l,:) = sum((repmat(data(l,:),nbins,1)>ref));
end
%%
%joint distributions
for l=1:L
for ll=1:L
	Joint{l}{ll} = zeros(nbins);
end
end

%%

for l=1:L
for n=1:nbins
	I=find(code(l,:)==n);
	for ll=l+1:L
		[Joint{l}{ll}(n,:)]=hist(data(ll,I),bins{ll});
	end
	
end
end
 

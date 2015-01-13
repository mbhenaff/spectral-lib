function [V, pools] = ms_spectral_clustering(w, wfat, depth, frac, psize)

W{1} = w;
WF{1} = wfat;

for j=1:depth
n = round(frac*size(W{j},1));
%1) spectral clustering 
fprintf('doing scale %d \n', j)
[V{j}, anchors, clusters] = spectral_clustering(W{j}, n, psize(j));
for i=1:n
slice = WF{j}(:,anchors(i));
[~,inds] = sort(slice,'descend');
pools{j}(:,i) = inds(1:psize(j));
end

%2) coarsen kernels
    nw=zeros(n);
    for s=1:n
        for t=1:n
            nw(s,t)=sum(sum(W{j}(clusters{s},clusters{t})));
        end    
    end
    W{j+1}=dn(nw,'ave');

    nw=zeros(n);
    for s=1:n
        for t=1:n
            nw(s,t)=sum(sum(WF{j}(pools{j}(:,s),pools{j}(:,t))));
        end    
    end
    WF{j+1}=dn(nw,'ave');
end



function [permut,invpermut,tmpP,W,V,P]=haarpartition_graph(W0,options)


%stage 1: fine to coarse

W{1} = W0;
V{1} = ones(size(W0,1),1);

maxscale=getoptions(options,'maxscale',0);

j=1;
while (size(W{j},1) > 1 & (maxscale==0 | j<= maxscale)) 
fprintf('%d..',size(W{j},1))
[W{j+1},V{j+1},P{j+1}] = AMGaggregate(W{j},options,V{j});
j=j+1;
end
fprintf('\n')

%stage 2: coarse to fine
J=size(P,2)-1;
for j=J:-1:1
[res,posits]=max(P{j+1},[],2);
tmpP{j}=posits;
end


%rearrange partitions to reflect the metric induced by the tree
qq=max(tmpP{J});
oldpermut=[1:qq];
for j=J:-1:1
	quants=max(tmpP{j});
	raster = 1;
	for n=1:quants
		F=find(tmpP{j}==oldpermut(n));
		for ff=F'
			permut(raster) = ff;
			invpermut(ff) = raster;
			raster = raster +1;
		end
	end
	oldpermut = permut;
end



function [W1,V1,P,C,F,anchors] = AMGaggregate(W0,options,V0)


if nargin < 3
		V0=ones(size(W0,1),1);
end

%eta=getoptions(options,'theta',2);
%Q=getoptions(options,'Q',0.4);
eta = 2;
Q = 0.4;

[sizex,res]=size(W0);
Wsum=sum(W0,2);


%compute future volume
Vf= V0 + (W0./(ones(sizex,1)*Wsum'))*V0;

%transfer to C the nodes with highest 
%future volume
pool = (Vf > eta * mean(Vf));


anchors=[];
%anchorsize=getoptions(options,'num_of_anchors',min(sizex/2,512));
anchorsize = min(sizex/2,512);

[res,visit]=sort(Vf,'descend');

C=find(pool==1);
F=find(pool==0);
for n=1:length(visit)
	if pool(visit(n))
			continue;
	end
	if sum(W0(visit(n),C))/Wsum(visit(n)) < Q
		pool(visit(n))=1;
		C=find(pool==1);
		if length(C) < anchorsize
			anchors = C;
		end
		F=find(pool==0);
	end
end

if length(C) == 0
min(W0(:))
max(W0(:))
sum(W0(visit(n),C))
Wsum(visit(n))
size(pool)
error('raro')
end

%build the coarser graph
%NN=getoptions(options,'NN',.5);
%Nmin=getoptions(options,'Nmin',max(1,round(length(C)*.25)));
Nmin=max(1,round(length(C)*0.25));


%[wrank,wpos]=sort(W0(:),'descend');
%wth = wrank(wpos(round(sizex*sizex*NN)));

P=zeros(sizex,length(C));
for f=F'
	slice=W0(f,C);
	[slisort,slipos]=sort(slice,'descend');
	neighb = slipos(1:min(length(slipos),Nmin));
	%neighb=find(W0(f,C)>wth);
	denom=sum(W0(f,C(neighb)));
	if(denom==0)
                length(C)
                length(neighb)
		error('coco')
	end
	for n=neighb'
		P(f,n) = W0(f,C(n)) / denom;
	end
end
P(C,:)=eye(length(C));

%define new volumes
V1 = ((V0')*P)';

%define new weights
W1=P'*(W0-diag(diag(W0)))*P;


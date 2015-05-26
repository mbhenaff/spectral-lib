if 0
close all
clear all

folder='/misc/vlgscratch3/LecunGroup/mbhenaff/merck/merck/paper/train/';

d=dir(folder);
maxfeat = 1;

for i=3:length(d)
 fname = d(i).name;
 if(length(fname)>4)
 if(strcmp( fname(end-3:end), '.csv')==1)
	code = csvread(fullfile(folder,'featurecode',fname));
	maxfeat = max(maxfeat,max(code(:)));
 end
end
end	
maxfeat

j1 = 32;
opts.kNN=j1;opts.alpha=1;opts.kNNdelta=j1;
bigkern = zeros(maxfeat,'single');
bigmass = zeros(maxfeat,'single');
for i=3:length(d)
 fname = d(i).name;
 if(length(fname)>4)
 if(strcmp( fname(end-3:end), '.csv')==1)
	aux = csvread(fullfile(folder,fname),1,2);
	code = csvread(fullfile(folder,'featurecode',fname));
	%normalize each feature
	auxn = sqrt(sum(aux.^2));
	aux = aux./repmat(auxn,size(aux,1),1);
	auxk = kernelization(aux');
	kersolo{i} = fgf_weights(auxk,opts);
	bigkern(code,code) = bigkern(code,code) + auxk;
	bigmass(code,code) = bigmass(code,code) + size(aux,1);
	fprintf('done dataset %s \n', fname)
 end
end
end

end

ker = bigkern./max(1,bigmass);

for i=3:length(d)
 fname = d(i).name;
 if(length(fname)>4)
 if(strcmp( fname(end-3:end), '.csv')==1)
	code = csvread(fullfile(folder,'featurecode',fname));
	kerb= ker(code,code);
	%codef{i} = code;
	kerf=fgf_weights(kerb,opts);	
	D = diag(sum(kerf).^(-1/2));
	L = eye(size(kerf,1)) - D * kerf * D;
	L = (L + L')/2;
	[ee,ev]=eig(L);
	save(fullfile(folder,sprintf('/graph_%d.mat',i)),'kerf','code','L','ee','ev','-v7.3');
end
end
end




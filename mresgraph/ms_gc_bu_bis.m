function [blocks fatblocks wout]=ms_gc_bu_bis(W,Wfat,ep,depth);
%function [blocks wout blocks_bi]=ms_gc_bu_dumb(W,ep,depth);
% multiscale graph cluster bottom up dumb
% don't use this on a big graph.

tw=W;
twfat=Wfat;
for k=1:depth
    wout{k}=tw;
    m=size(tw,1);    
    count=0;
    lss=m;
    sms=zeros(m,1);
    bid=[1:m];
    TT=[];
    while lss>0               
        id=ceil(lss*rand);
        id=bid(id);
        nzid=find(tw(:,id));
        tv=tw(nzid,id);
        uid=find(tv>ep(k));
        stv=tv(uid);
        uid=nzid(uid); 
        sms(uid)=max(sms(uid),stv);
        sms(id)=1;
        uid=union(uid,id);
        count=count+1;
        blocks{k}{count}=uid;
        aux = zeros(m,1);
        aux(uid) = 1;
        TT=[TT aux];
        bid=find(sms<ep(k)); 
        lss=length(bid);              
    end
    nblocks=length(blocks{k});
    %compute the 'fat blocks'
    tempo = twfat*TT;
    [~,centers]=max(tempo);
    for s=1:length(centers)
        fatblocks{k}{s}=find(twfat(centers(s),:));
    end
    %building a full matrix here....
    nw=zeros(nblocks);
    for s=1:nblocks
        for t=1:nblocks
            nw(s,t)=sum(sum(tw(blocks{k}{s},blocks{k}{t})));
        end    
    end
    tw=dn(nw,'ave');
    %building a full matrix here....
    nw=zeros(nblocks);
    for s=1:nblocks
        for t=1:nblocks
            nw(s,t)=sum(sum(twfat(blocks{k}{s},blocks{k}{t})));
        end    
    end
    twfat=dn(nw,'ave');

end

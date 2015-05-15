#!/bin/bash


nrunning=`ls -1 jobs/running/ | wc -l`
nqueued=`ls -1 jobs/queued/ | wc -l`
cntr=$((nrunning+nqueued))
cntr=$((cntr+1))

while read p; do 
    echo $p > jobs/queued/job$cntr.txt
    cntr=$((cntr+1))
done < jobs.txt

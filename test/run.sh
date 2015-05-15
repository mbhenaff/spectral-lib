#!/bin/bash 

while :
do

    if ls -1 jobs/queued/* > /dev/null 2>&1 
    then
        for job in jobs/queued/*.txt; do 
            cmd=`cat $job`
            fname=$(basename $job)
            rm -f $job
            echo $cmd > jobs/running/$fname
            `$cmd -gpunum $1`
            echo $cmd > jobs/finished/$fname           
            rm -f jobs/running/$fname
            break
        done
    else
        echo queue is empty!
        exit 1;
    fi
done
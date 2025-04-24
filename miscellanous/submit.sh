#!/bin/bash
# parameters: jobname jobtime_in_hours sleep_time
for ((i = 0; i <20; i++))
do
	echo $i
	sbatch jobs/job${i}.sh
	sleep 5
done

#!/bin/bash 

nohup ./run_training.sh >  no_hup.log 2>&1 &
echo $! > save_pid.txt

echo nohup job created!


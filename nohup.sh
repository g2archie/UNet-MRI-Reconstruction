#!/bin/bash 

nohup ./train.sh >  no_hup.log 2>&1 &

echo nohup job created!


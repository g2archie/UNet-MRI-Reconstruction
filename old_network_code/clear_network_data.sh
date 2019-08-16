#!/bin/bash
#
# Following script will clear all data associated with trained network.
#

NETWORK_DATA_DIR=./network

read -p "Are you sure? " -n 1 -r
echo    
if [[ ! $REPLY =~ ^[Yy]$ ]]
    rm -r $NETWORK_DATA_DIR
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 
fi

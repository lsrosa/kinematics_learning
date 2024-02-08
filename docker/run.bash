#!/bin/bash

DP=X11
if [ "$1" == "nvidia" ]; then
  DP=nvidia
fi

cd ..
docker-compose -f  docker/all_$DP.yml up $2


#!/bin/bash

UPAR="--build-arg UID=`id -u` --build-arg GID=`id -g`"


DEV="cpu"
if [ "$1" != "" ]; then
  DEV="$1"
fi

DOCKERFILE=Dockerfile.ml-$DEV

IMAGENAME=torch_gym_sb3:$DEV


echo "====================================="
echo "   Building $IMAGENAME  "
echo "====================================="

docker build $UPAR -t $IMAGENAME -f $DOCKERFILE .

if [ "$DEV" != "cpu" ]; then
  docker tag  $IMAGENAME torch_gym_sb3:gpu
fi


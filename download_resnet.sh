#! /usr/bin/env bash
RESNET_MODEL=$(echo $1 | awk '{print toupper($0)}')

CHECKPOINTS_DIR=resnet/checkpoints
mkdir -p $CHECKPOINTS_DIR

if [ -z $RESNET_MODEL ] || [ $RESNET_MODEL == 'V2_101'  ]; then
    echo 'Downloading resnet_v2_101 checkpoint...'
    DL_LINK='http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz'
elif [ $RESNET_MODEL == 'V2_50' ]; then
    echo 'Downloading resnet_v2_50 checkpoint...'
    DL_LINK='http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz'
fi

wget -O- $DL_LINK | tar xvz -C $CHECKPOINTS_DIR

echo 'Done.'

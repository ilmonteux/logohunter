#! /bin/bash
echo "Hello $USER"
echo "This script will set up the logohunter environment:"
echo " - downloading pretrained YOLOv3 weights..."

# move to src/keras_yolo3, download YOLO weigths
cd $(dirname $0)/../src/keras_yolo3/

echo "Downloading https://logohunters3.s3-us-west-2.amazonaws.com/yolo_weights_logos.h5 in src/keras_yolo3"
wget logohunters3.s3-us-west-2.amazonaws.com/yolo_weights_logos.h5

# move to src, download pre-computed logo features
cd ../
echo "Downloading https://logohunters3.s3-us-west-2.amazonaws.com/inception_logo_features_200_trunc2.hdf5 in src/"
wget logohunters3.s3-us-west-2.amazonaws.com/inception_logo_features_200_trunc2.hdf5

echo "Downloading https://logohunters3.s3-us-west-2.amazonaws.com/vgg16_logo_features_128.hdf5 in src/"
wget logohunters3.s3-us-west-2.amazonaws.com/vgg16_logo_features_128.hdf5

#! /bin/bash
echo "Hello $USER"
echo "This script will set up the logohunter environment:"
echo " - downloading pretrained YOLOv3 weights..."

# move to src/keras_yolo3, download YOLO weigths
cd $(dirname $0)/../src/keras_yolo3/

echo "wget https://logohunters3.s3-us-west-2.amazonaws.com/yolo_weights_logos.h5"
# wget logohunters3.s3-us-west-2.amazonaws.com/yolo_weights_logos.h5

# move to src, download logo instances
cd ../
echo "wget https://logohunters3.s3-us-west-2.amazonaws.com/inception_logo_features.hdf5"
# wget logohunters3.s3-us-west-2.amazonaws.com/inception_logo_features.hdf5

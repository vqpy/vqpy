#!/bin/bash

VQPY_ROOT=$(cd $(dirname $0) ; pwd)
#echo $VQPY_ROOT
PROPERTY_LIB_DIR="$(cd ${VQPY_ROOT}/vqpy/property_lib; pwd)"
#echo $PROPERTY_LIB_DIR

vehicle=${1:-true}

if [ "$vehicle" == "true" ]; then
  echo "Downloading vehicle models."
  cd $PROPERTY_LIB_DIR/vehicle/models
  echo "Downloading licence plate detection models into $PWD."
  git clone https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git
  mv License_Plate_Detection_Pytorch lpdetect
fi
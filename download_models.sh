#!/bin/bash

VQPY_SRC_ROOT=$(cd $(dirname $0)/vqpy ; pwd)
#echo $VQPY_SRC_ROOT
DETECTOR_MODEL_DIR="$(cd ${VQPY_SRC_ROOT}/operator/detector/models; pwd)"
#echo $DETECTOR_MODEL_DIR
DETECTOR_WEIGHTS_DIR="${VQPY_SRC_ROOT}/operator/detector/weights"
mkdir -p $DETECTOR_WEIGHTS_DIR
#echo $DETECTOR_WEIGHTS_DIR
PROPERTY_LIB_DIR="$(cd ${VQPY_SRC_ROOT}/property_lib; pwd)"
#echo $PROPERTY_LIB_DIR

# install yolox
if python -c "import yolox" &> /dev/null; then
  echo "Found installed yolox."
else
  cd $DETECTOR_MODEL_DIR/torch
  if [ -f $PWD/YOLOX/setup.py ]; then
    echo "Found cloned YOLOX repo."
  else
    echo "Cloning YOLOX into $PWD"
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
  fi
  echo "Install yolox."
  cd YOLOX
  pip3 install -v -e .
fi

cd $DETECTOR_WEIGHTS_DIR
if [ ! -f $PWD/yolox_x.pth ]; then
  echo "Downloading yolox weights into $PWD"
  wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
fi

vehicle=${1:-true}

if [ "$vehicle" == "true" ]; then
  echo "Vehicle models will be prepared."
  # download license plate model LPR
  cd $PROPERTY_LIB_DIR/vehicle/models
  LICENCE_PLATE_MODEL_DIR="lpdetect"
  if [ ! -f $PWD/${LICENCE_PLATE_MODEL_DIR}/main.py ]; then
    echo "Downloading licence plate detection model (LPR) into $PWD."
    git clone https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git
    mv License_Plate_Detection_Pytorch $LICENCE_PLATE_MODEL_DIR
  else
    echo "Licence plate detection model (LPR) exists. Skip download."
  fi
  # install license plate model openalpr
  if python -c "import openalpr" &> /dev/null; then
    echo "Found installed openalpr."
  else
    pip install openalpr
    "Openalpr is installed."
  fi
fi

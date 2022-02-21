#!/bin/bash

# Set non interactive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

if [[ $OSTYPE == "darwin"* ]]
then
  brew install gcc git cmake
  brew install llvm
elif [[ "$(grep '^ID_LIKE' /etc/os-release)" == *"centos"* ]]
then
  sudo yum update -y && sudo yum install -y gcc gcc-c++ llvm-devel cmake3 libxml2-dev
  if [ -f "/usr/bin/cmake" ]
  then
    sudo alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake 10 \
      --slave /usr/local/bin/ctest ctest /usr/bin/ctest \
      --slave /usr/local/bin/cpack cpack /usr/bin/cpack \
      --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake \
      --family cmake
    sudo alternatives --install /usr/local/bin/cmake cmake /usr/bin/cmake3 20 \
      --slave /usr/local/bin/ctest ctest /usr/bin/ctest3 \
      --slave /usr/local/bin/cpack cpack /usr/bin/cpack3 \
      --slave /usr/local/bin/ccmake ccmake /usr/bin/ccmake3 \
      --family cmake
  else
    sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
  fi
else
  apt-get update && apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
fi

if [ ! -d "tvm" ]
then
  git clone --recursive https://github.com/apache/tvm tvm
fi

cd tvm
mkdir -p build
cp $CONFIG_PATH build/
cd build
cmake ..
make -j8
cd ../python
python3 setup.py install --user
cd ../..
if [[ $OSTYPE == "darwin"* ]]
then
  brew install openblas gfortran
  pip install pybind11 cython pythran
  conda install -y scipy
  pip install xgboost
else
  pip3 install decorator attrs tornado psutil xgboost cloudpickle
fi
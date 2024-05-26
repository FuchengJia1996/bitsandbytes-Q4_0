#!/bin/bash

#export CUDA_HOME=${HOME}/Library/cuda-12.4
#export PATH=${CUDA_HOME}/bin:$PATH
#export CPATH="${CUDA_HOME}/include"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64

#cmake -DCOMPUTE_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=80 -S .

cmake -DCOMPUTE_BACKEND=cuda -S .

make

#!/bin/bash

#export CUDA_HOME=${HOME}/Library/cuda-12.4
#export PATH=${CUDA_HOME}/bin:$PATH
#export CPATH="${CUDA_HOME}/include"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64
export BITSANDBYTES_HOME=$(pwd)/bitsandbytes
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BITSANDBYTES_HOME}

python examples/quantize_4bit.py

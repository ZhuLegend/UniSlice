#! /bin/bash
nvidia-smi -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
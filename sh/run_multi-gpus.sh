#!/bin/bash

# Set the number of GPUs to use
NUM_GPUS=5

# Launch a separate instance of the script for each GPU
for ((i=0; i<$NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup bash run_experiment.sh $i > log$i.out 2>&1 & 
    sleep 5 &
done


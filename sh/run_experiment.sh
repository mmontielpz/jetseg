#!/bin/bash

# Set the GPU ID
export CUDA_VISIBLE_DEVICES=$1

START_TIME=$(date +%s)

# Setting directories
PYTHON_DIR="${PWD}/../py"
RESULTS_DIR="${PWD}/../results"
TRAIN_DIR="${PWD}/../train"
EXPERIMENTS_DIR="${PWD}/../experiments"

#Setting experiment
DATE="$(date +"%d-%m-%Y")"
TIME="$(date +"%R")"
EXPERIMENT_NAME="experiment-${DATE}-${TIME}-GPU-ID-${CUDA_VISIBLE_DEVICES}"
echo "Experiment Name ${EXPERIMENT_NAME}"

EXPERIMENT_DIR="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}"

# Check if experiment directory exists
if ! [ -d "${EXPERIMENTS_DIR}" ]
then
    mkdir $EXPERIMENTS_DIR
fi

# Check if experiment exists
if [ -d "${EXPERIMENT_DIR}" ]
then
    # Create experiment directory
    echo "Experiment $EXPERIMENT_DIR exists!"
    exit
else
    mkdir $EXPERIMENT_DIR
fi

# Run train, test and evaluation for semantic segmentation
export CUDA_VISIBLE_DEVICES=$1
python3 $PYTHON_DIR/train.py --jsc=$1 --num_epochs=1&&
export CUDA_VISIBLE_DEVICES=$1
python3 $PYTHON_DIR/test.py --jsc=$1 &&
export CUDA_VISIBLE_DEVICES=$1
python3 $PYTHON_DIR/eval.py --jsc=$1 &&
export CUDA_VISIBLE_DEVICES=$1
python3 $PYTHON_DIR/perf.py --jsc=$1 &&

echo "Total time elapsed: $(date -ud "@$(($(date +%s) - $START_TIME))" +%T) (HH:MM:SS)"

echo "Total time elapsed: $(date -ud "@$(($(date +%s) - $START_TIME))" +%T) (HH:MM:SS)" > $RESULTS_DIR/time.txt

# Copy results to experiment directory
cp -r $RESULTS_DIR $EXPERIMENT_DIR
cp -r $TRAIN_DIR $EXPERIMENT_DIR

echo  "Finished experiment!!"

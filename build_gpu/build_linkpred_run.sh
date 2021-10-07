#!/bin/bash

###############################
# TODO:
#    1. Set dataset name using DATASET variable.
###############################

# TODO: remove the following line if you do not want to clean and build from scratch every time.
make clean
rm CMakeCache.txt

PARAMS_FILE_DIR=params_files

# This will work in the provided docker environment.
echo "Compiling..."
cp ../CMakeLists_linkpred_gpu.txt ../CMakeLists.txt
#TODO: add your libtorch path below...
cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/ ..
cmake --build . --config Release

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

#TODO: add dataset name below...
DATASET='dataset-name'

ALGO='link-prediction'

echo "Running..."

for THIS_DATASET in ${DATASET}
do
    echo "Executing linkpred for ${THIS_DATASET}"
    ./${ALGO} -f ../data/link_pred/${THIS_DATASET}.wel -c ${PARAMS_FILE_DIR}/linkpred_params.txt
done

# To run the GPU profiler:

# CUDA_VISIBLE_DEVICES=0 \
#    nv-nsight-cu-cli \
#    --section InstructionStats \
#    --section MemoryWorkloadAnalysis \
#    --section Occupancy \
#    --section SpeedOfLight \
#    ./${ALGO} -f ../data/link_pred/${THIS_DATASET}.wel -c ${PARAMS_FILE_DIR}/linkpred_params.txt
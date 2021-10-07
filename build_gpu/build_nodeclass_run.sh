#!/bin/bash

###############################
# TODO:
#    1. Set dataset name (same as dataset directory name) using DATASET variable.
###############################

# TODO: remove the following line if you do not want to clean and build from scratch every time.
make clean
rm CMakeCache.txt

PARAMS_FILE_DIR=params_files

# This will work in the provided docker environment.
echo "Compiling..."
cp ../CMakeLists_nodeclass_gpu.txt ../CMakeLists.txt
#TODO: add your libtorch path below...
cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.8/site-packages/torch/ ..
cmake --build . --config Release

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

#TODO: add dataset name below...
DATASET='dataset-directory-name' # only the name of the dataset directory in data/ is required.
# See data/dblp3 for more details

ALGO='node-classification'

echo "Running..."

for THIS_DATASET in ${DATASET}
do
    echo "Running ./${ALGO} -f ../data/node_class/${THIS_DATASET}/tgraph.wel -c ../params_files/nodeclass_params.txt -p ${THIS_DATASET}"
    ./${ALGO} -f ../data/node_class/${THIS_DATASET}/tgraph.wel -c ${PARAMS_FILE_DIR}/nodeclass_params.txt -p ${THIS_DATASET}
done

# To run the GPU profiler:

# CUDA_VISIBLE_DEVICES=0 \
#    nv-nsight-cu-cli \
#    --section InstructionStats \
#    --section MemoryWorkloadAnalysis \
#    --section Occupancy \
#    --section SpeedOfLight \
#    ./${ALGO} -f ../data/node_class/${THIS_DATASET}/tgraph.wel -c ${PARAMS_FILE_DIR}/nodeclass_params.txt -p ${THIS_DATASET}
A Deep Dive Into Understanding The Random Walk-Based Temporal Graph Learning
===================

This directory contains implementations for two key temporal graph representation learning kernels: (a) link prediction (linkpred) and (b) node classification (nodeclass).  
This paper was presented at 2021 IEEE International Symposium on Workload Characterization (IISWC 2021).   

If you find this repo useful, please cite the following paper:
```
@inproceedings{talati2021temp,
  author    = {Nishil Talati and
               Di Jin and
               Haojie Ye and
               Ajay Brahmakshatriya and
               Ganesh Dasika and
               Saman Amarasinghe and
               Trevor Mudge and
               Danai Koutra and
               Ronald Dreslinski},
  title     = {A Deep Dive Into Understanding The Random Walk-Based Temporal Graph Learning},
  booktitle = {{IEEE International Symposium on Workload Characterization (IISWC) 2021, Virtual, November 7-9, 2021}},
  pages     = {},
  publisher = {IEEE},
  year      = {2021}
}
```

Preparing Datasets:
-----------

**Input Arguments.**

The expected input dataset file extension for running linkpred/nodeclass is '.wel'.
Each row of the file follows this format:
\<node1 node2 timestamp\>

Additionally, a configuration script is used for setting parameters.
This is passed to the command line using ```-c``` flag. 
Example scripts are provided in ```build_cpu/params_files``` directory.

The linkpred algorith only requires one graph file - use ```-f``` flag to set the path of the input file.
Instructions to download datasets and prepare temporal graph files for link prediction are present in ```data/link_pred/``` folder.
In addition to real-world datasets, this directory also contains a file to generate a synthetic dataset.
The instructions to generate this are present in the README file in ```data/link_pred/```.

The nodeclass algorithm requires four files -   
    1) graph file (tgrph.wel),    
    2) labeled training dataset (train.tsv) -- file format: \<node label\>    
    3) labeled validation dataset (valid.tsv) -- file format: \<node label\>    
    4) labeled testing dataset (test.tsv) -- file format: \<node label\>    
These files are expected to be present in ```data/node_class/#dataset-name#``` folder.    
See ```data/node_class/dblp3``` for more instructions.

To set the dataset in provided biuld scripts (as described below), set the ```DATASET``` flags in build files (search for ```TODO``` in build scripts).

List of Dependencies:
-----------

```
cmake
gcc
openmp
nvcc (docker has this dependency resolved)
mkl/blis
libtorch
argparse
networkx
numpy
tqdm
sklearn
```

For CPU:
-----------

**Build Instructions.**

Linking with LibTorch:
Follow instructions on [THIS](https://pytorch.org/cppdocs/installing.html) website to download the torch library.
Note that this might not be the most optimized library for AMD CPUs. Compile PyTorch from source using BLIS library and link that to get the most optimized version of PyTorch for AMD CPUs.

Both applications are built independely.
```CMakeLists_linkpred_cpu.txt``` contains instructions to build link prediction, ```CMakeLists_nodeclass_cpu.txt``` contains instructions to build node classification.
To build individual individual kernels, copy ```CMakeLists_{linkpred/nodeclass}_cpu.txt``` to ```CMakeLists.txt```.
Note that the build script provided in ```build_cpu/``` directory already does this for you, so you need not do this manually (unless you write your own algorithm).



**Run Instructions.**

Once you have (a) downloaded/installed libtorch and (b) prepared input datasets, follow these instructions to run the benchmarks.
Sample scripts to run the benchmarks are present in the ```build_cpu/``` directory.
For example, to run link prediction, run the following file: ```build_linkpred_run.sh```.
Be sure to change the the libtorch path for the build system to find torch files using ```-DCMAKE_PREFIX_PATH```.
Make sure that the input datasets are present in ```data/link_pred``` or ```data/node_class/dataset-name``` folders.
After doing all of the above, the benchmarks can be compiled and run by using one script: 

```cd build_cpu && ./build_linkpred_run.sh``` or ```cd build_cpu && ./build_nodeclass_run.sh```.
The scripts need to run from within the ```build_cpu``` directory as they contain relative paths to dataset files from there.


For GPU:
-----------

We provide a docker image to run the applications on a GPU. Use the following command to invoke the docker image:

```
# TODO: change /this-directory-path/ to the path of this directory in your system...
sudo docker run --cap-add SYS_ADMIN --security-opt seccomp=unconfined --gpus all -it --rm --shm-size 8G -v /this-directory-path/:/rwalk rwalklearn/iiswc21
```

This code will be present at ```/rwalk``` directory inside the docker.
The path for linking libtorch inside the docker is already present in ```build_gpu/build_{linkpred/nodeclass}_run.sh``` file.

After doing all of the above for GPU, the benchmark can be compiled and run with the following script:

```cd build_gpu && ./build_linkpred_run.sh``` or ```cd build_gpu && ./build_nodeclass_run.sh```.
The scripts need to run from within the ```build_gpu``` directory as they contain relative paths to dataset files from there.

To run the applications with NVIDIA profiler and gather various statistics, checkout the commented lines at the end of ```build_gpu/build_linkpred_run.sh``` or ```build_gpu/build_nodeclass_run.sh``` script.
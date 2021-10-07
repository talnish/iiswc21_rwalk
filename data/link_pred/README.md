Setup data files for link prediction
===================

Real-word Temporal Graphs
-----------

Download datasets from [HERE](http://networkrepository.com/dynamic.php) or [HERE](http://snap.stanford.edu/data/index.html#temporal)

For example, download wiki-talk temporal graph using the following command:     
Ia-email: ```wget https://nrvis.com/download/data/dynamic/ia-enron-email-dynamic.zip```    
Wiki-talk: ```wget http://snap.stanford.edu/data/wiki-talk-temporal.txt.gz```  
Stackoverflow: ```wget http://snap.stanford.edu/data/sx-stackoverflow.txt.gz```     

Unzip the files (using ```unzip/gunzip```), and pre-process datasets before using them.  
Before preprocessing the datasets, make sure that the datasets only contain graph edgelist.  
If there are text-based lines starting from "#", remove them.  

An example commands to pre-process datasets is the following:
```
$ python preprocess_dataset.py -i dataset-filename.fileextension
```

After preprocessing, you should have datasets with three columns as follows
```
src_node dst_node norm_timestamp
```
Here, norm_timestamp is the normalized timestamp between 0 and 1.
These datasets can be used for link prediction.

Synthetic Temporal Graphs
-----------

To generate synthetic graphs, use the following script:
```
python generate_synthetic.py -n #nodes# -e #edges# -s #seed#
```
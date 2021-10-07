Setup data files for node classification
===================

Real-world Temporal Graphs
-----------

Download datasets from [HERE](https://www.dropbox.com/s/0h6b5p5uw63pjbb/STAR-Code-Data.zip?dl=0&file_subpath=%2FSTAR-Code-Data%2FData) and place it in this directory.

Use ```process_dataset.py``` script to generate a temporal graph, and training/testing/validation datasets as follows.
```
python process_dataset.py -i DBLP3.npz
```

This will create four files: (a) tgraph.wel, (b) train.tsv, (c) valid.tsv, and (d) test.tsv.     
All of these files are used in running node classification. Note that the script to run node classification (i.e., ```build_nodeclass_run.sh```) only needs the name of this directory in the ```DATASET``` field. Using this name, the four required files will be automatically obtained by the script.

To use another dataset, create a directory within ```data/``` with the name of the dataset, and copy ```process_dataset.py``` file.
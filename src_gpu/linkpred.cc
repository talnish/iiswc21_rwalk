#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <fstream>
#include <experimental/algorithm>
#include <torch/torch.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
// #include "graph.h"
#include "pvector.h"
#include "timer.h"

typedef NodeWeight<NodeID, WeightT> WNode;
typedef EdgePair<NodeID, WNode> EdgeP;
typedef pvector<EdgeP> EdgeList;
typedef std::map<int, std::vector<float>> NodeEmb;
typedef std::vector<std::pair<NodeID, WeightT>> TempNodeVector;
typedef std::pair<NodeID, WeightT> TNode;
typedef std::vector<double> DoubleVector;

// Temporal edge list
struct TempELStruct {
  WeightT time_stamp;
  NodeID src_node;
  NodeID dst_node;
};

// Edge pairs
struct EdgePairStruct {
  NodeID src_node;
  NodeID dst_node;
};

static std::random_device rd;
static std::mt19937 rng(rd());


// CAUTION: This will print the entire training/testing datasets
//          Can fill up the terminal and slow down the program!
bool print_datasets = false;

#if defined(CILK)
#include <cilk/cilk.h>
#include <cilk/reducer_list.h>
#include <list>
#define parallel_for cilk_for
#elif defined(OPENMP)
#include <omp.h>
// #define parallel_for _Pragma("omp parallel for") for
#define parallel_for _Pragma("omp parallel for schedule(dynamic, 64)") for
// #define parallel_for _Pragma("omp parallel for schedule(guided)") for
#else
#define parallel_for for
#endif

#include "rwalk.h"
#include "word2vec.h"
#include "linkpred_datapreproc_opt.h"
#include "linkpred_model.h"
#include "linkpred_dataloader.h"
#include "linkpred_classifier.h"

/*
  Link prediction on a temporal graph.
  Input arguments
  @ -f graph-filename.wel 
  @+ input file for a temporal network 
  @+ format: <src_node dst_node timestamp>
  
  @ -c cofig-filename.txt
  @+ configuration file to set parameters
  @+ see example file - build/linkpred_params.txt
*/

int main(int argc, char* argv[]) {

  CLApp cli(argc, argv, "link-prediction");

  if (!cli.ParseArgs())
    return -1;

  // Data structures
  WeightedBuilder b(cli);
  EdgeList el;
  WGraph g = b.MakeGraph(&el);
  NodeEmb node_emb;

  // Read parameter configuration file
  cli.read_params_file();

  // Parameter initialization
  int   max_walk_length     =   cli.get_max_walk_length();
  float ratio               =   cli.get_training_ratio();
  int   node_embedding_dim  =   cli.get_node_emb_dim();
  int   num_walks_per_node  =   cli.get_num_walks_per_node();
  int   num_workers         =   cli.get_num_workers();
  int   output_dim          =   cli.get_output_dim();
  float learning_rate       =   cli.get_learning_rate();
  int   num_epochs          =   cli.get_num_epochs();
  int   hidden_layer_dim    =   cli.get_hidden_layer_dim();
  int   batch_size          =   cli.get_batch_size();
  float target_accuracy     =   cli.get_target_val_accuracy();

  // Number of threads
  int num_threads;
  if(cli.use_max_num_threads() == 1) num_threads = omp_get_max_threads();
  else num_threads = cli.get_num_threads();
  omp_set_num_threads(num_threads);
  printf("Using %d thread(s) for running.\n", num_threads);

  // Parameter printing
  std::cout << "\n---- PARAM VALUES ----\n";
  std::cout << "num_threads         : " << num_threads << std::endl;
  std::cout << "num_walks_per_node  : " << num_walks_per_node << std::endl;
  std::cout << "max_walk_length     : " << max_walk_length << std::endl;
  std::cout << "node_embedding_dim  : " << node_embedding_dim << std::endl;
  std::cout << "num_workers         : " << num_workers << std::endl;
  std::cout << "learning_rate       : " << learning_rate << std::endl;
  std::cout << "num_epochs          : " << num_epochs << std::endl;
  std::cout << "hidden_layer_dim    : " << hidden_layer_dim << std::endl;
  std::cout << "batch_size          : " << batch_size << std::endl;
  std::cout << "target_accuracy     : " << target_accuracy << std::endl;

  // Initialize arrays
  long long int test_dataset_size = g.num_edges() * (1 - ratio);
  // Assuming the ratio to be 0.2, 
  //    if other, change the number 3 to an appropriate choice
  long long int train_dataset_size = 3 * test_dataset_size;
  long long int valid_dataset_size = test_dataset_size;
  TempELStruct*   temp_el      = new TempELStruct[g.num_edges()];
  EdgePairStruct* train_p_list = new EdgePairStruct[train_dataset_size];
  EdgePairStruct* train_n_list = new EdgePairStruct[train_dataset_size];
  EdgePairStruct* test_p_list  = new EdgePairStruct[test_dataset_size];
  EdgePairStruct* test_n_list  = new EdgePairStruct[test_dataset_size];
  EdgePairStruct* valid_p_list = new EdgePairStruct[valid_dataset_size];
  EdgePairStruct* valid_n_list = new EdgePairStruct[valid_dataset_size];

  // Compute temporal random walk
  std::cout << "\n---- RWALK ----\n";
  compute_random_walk(
    /* temporal graph */ g, 
    /* max random walk length */ max_walk_length,
    /* number of rwalks/node */ num_walks_per_node,
    /* filename of random walk */ "out_random_walk.txt"
  );

  // Call word2vec function to create node embeddings
  std::cout << "\n---- WORD2VEC ----\n";
  custom_word2vec(
    /* node embedding map */ &node_emb,
    /* train_file */ "out_random_walk.txt",   
    /* output_file */ "node_emb.txt",         
    /* layer1_size */ node_embedding_dim,     
    /* min_cnt */ 0,
    /* window */ 10,
    /* iter */ 1, 
    /* cbow */ 0, // skip-gram model
    /* num_threads */ num_threads,
    /* print embedding to a file */ print_datasets
  );

  // Data pre-processing step to create dataset for the classifier
  std::cout << "\n---- PREPROC ----\n";
  link_prediction_data_preprocessing(
    /* temporal graph */ g, 
    /* edge list */ el,
    /* temporal edge list */ temp_el,
    /* positive samples for training */ train_p_list, 
    /* negative samples for training */ train_n_list, 
    /* positive samples for testing  */ test_p_list, 
    /* negative samples for testing */ test_n_list,
    /* positive samples for validation */ valid_p_list,
    /* positive samples for validation */ valid_n_list,
    // /* ratio of dataset division */ ratio       
    /* num samples in training dataset */ train_dataset_size,
    /* num samples in testing dataset */ test_dataset_size,
    /* num samples in validation dataset */ valid_dataset_size
  );
  delete[] temp_el;

  // Link prediction classifier that performs both training and testing
  std::cout << "\n---- CLASSIFIER ----\n";
  link_prediction_clasifier(
    /* temporal graph */ g, 
    /* positive samples for training */ train_p_list, 
    /* negative samples for training */ train_n_list, 
    /* positive samples for testing  */ test_p_list, 
    /* nositive samples for testing */ test_n_list,
    /* positive samples for validation  */ valid_p_list, 
    /* nositive samples for validation */ valid_n_list,
    /* node embeddings */ node_emb,
    /* node embedding dimension */ node_embedding_dim,
    // /* dataset size of training/testing */ train_test_dataset_size,
    /* num samples in training dataset */ train_dataset_size,
    /* num samples in testing dataset */ test_dataset_size,
    /* num samples in validation dataset */ valid_dataset_size,
    /* output dim of logistic regression */ output_dim,
    /* learning rate */ learning_rate,
    /* number of epochs for training */ num_epochs,
    /* number of neurons in the hidden layer */ hidden_layer_dim,
    /* batch_size */ batch_size,
    /* target validation accuracy */ target_accuracy,
    /* number of threads */ num_threads,
    /* number of workers for parallel data loader */ num_workers
  );

  return 0;
}
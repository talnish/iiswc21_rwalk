#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <map>
#include <set>
#include <fstream>
#include <experimental/algorithm>
#include <torch/torch.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

typedef NodeWeight<NodeID, WeightT> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;
typedef std::tuple<NodeID, NodeID, WeightT> EdgeTuple;
typedef std::vector<EdgeTuple> TempEL;
typedef std::pair<NodeID, NodeID> NodePair;
typedef std::vector<NodePair> ELPair;
typedef std::set<NodeID> NodeSet;
typedef std::map<int, std::vector<float>> NodeEmb;
typedef std::vector<NodeID> NodeVector;
typedef std::vector<std::vector<NodeID>> NodeVectorSet;
typedef std::vector<std::pair<NodeID, WeightT>> TempNodeVector;
typedef std::pair<NodeID, WeightT> TNode;
typedef std::vector<double> DoubleVector;
typedef std::vector<std::pair<NodeID, long int>> LabeledNode;

struct InputDataSize {
  int training_data_size;
  int validation_data_size;
  int testing_data_size;
};

struct LabeledData {
  NodeID node_id;
  int node_label;
};

static std::random_device rd;
static std::mt19937 rng(rd());

// Print datasets for debugging?
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
#define parallel_for _Pragma("omp parallel for") for
#else
#define parallel_for for
#endif

#include "rwalk.h"
#include "word2vec.h"
#include "nodeclass_dataloader.h"
#include "nodeclass_model.h"
#include "nodeclass_classifier.h"
#include "nodeclass_datapreproc.h"

/*
  Node classification on a temporal graph.
  Input arguments
  @ -f filename1.wel 
  @+ input file for a temporal network 
  @+ format: <src_node dst_node timestamp>

  @ -p name of the dataset
  @+ see build/build_nodeclass_run.sh to understand 
  @+ how the benchmark is run
  
  @ -c cofig-filename.txt
  @+ configuration file to set parameters
  @+ see example file - build/nodeclass_params.txt
*/

int main(int argc, char* argv[]) {

  CLApp cli(argc, argv, "node-classification");

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
  int   node_embedding_dim  =   cli.get_node_emb_dim();
  int   num_walks_per_node  =   cli.get_num_walks_per_node();
  int   num_workers         =   cli.get_num_workers();
  float learning_rate       =   cli.get_learning_rate();
  int   num_epochs          =   cli.get_num_epochs();
  int   hidden_layer1_dim   =   cli.get_hidden_layer1_dim();
  int   hidden_layer2_dim   =   cli.get_hidden_layer2_dim();
  int   batch_size          =   cli.get_batch_size();
  float target_accuracy     =   cli.get_target_val_accuracy();
  
  std::string training_file_path = cli.get_training_file_name();
  std::string validation_file_path = cli.get_validation_file_name();
  std::string testing_file_path = cli.get_testing_file_name();

  // Number of threads
  int num_threads;
  if(cli.use_max_num_threads() == 1) num_threads = omp_get_max_threads();
  else num_threads = cli.get_num_threads();
  omp_set_num_threads(num_threads);
  // torch::set_num_threads(num_threads);
  printf("Using %d thread(s) for running.\n", num_threads);

  // Parameter printing
  std::cout << "\n---- PARAM VALUES ----\n";
  std::cout << "num_threads           : " << num_threads << std::endl;
  std::cout << "num_walks_per_node    : " << num_walks_per_node << std::endl;
  std::cout << "max_walk_length       : " << max_walk_length << std::endl;
  std::cout << "node_embedding_dim    : " << node_embedding_dim << std::endl;
  std::cout << "num_workers           : " << num_workers << std::endl;
  std::cout << "learning_rate         : " << learning_rate << std::endl;
  std::cout << "num_epochs            : " << num_epochs << std::endl;
  std::cout << "hidden_layer1_dim     : " << hidden_layer1_dim << std::endl;
  std::cout << "hidden_layer2_dim     : " << hidden_layer2_dim << std::endl;
  std::cout << "batch_size            : " << batch_size << std::endl;
  std::cout << "target_accuracy       : " << target_accuracy << std::endl;
  std::cout << "training_file_path    : " << training_file_path << std::endl;
  std::cout << "validation_file_path  : " << validation_file_path << std::endl;
  std::cout << "testing_file_path     : " << testing_file_path << std::endl;

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
    /* train_file */ "out_random_walk.txt",   // TODO: remove this
    /* output_file */ "node_emb.txt",         // TODO: remove this
    /* layer1_size */ node_embedding_dim,     // TODO: pass it from the command line
    /* min_cnt */ 0,
    /* window */ 10,
    /* iter */ 1, 
    /* cbow */ 0, // skip-gram model
    /* num_threads */ num_threads,
    /* print embedding to a file */ false //print_datasets
  );

  // Find the size of training/testing data
  InputDataSize in_data_size;
  std::ifstream train_input(training_file_path);
  if(train_input)
  {
    int cnt = 0;
    std::string in_line;
    while(getline(train_input, in_line))
    {
      cnt++;
    }
    in_data_size.training_data_size = cnt;
  }
  train_input.close();
  
  std::ifstream valid_input(validation_file_path);
  if(valid_input)
  {
    int cnt = 0;
    std::string in_line;
    while(getline(valid_input, in_line))
    {
      cnt++;
    }
    in_data_size.validation_data_size = cnt;
  }
  valid_input.close();

  std::ifstream test_input(testing_file_path);
  if(test_input)
  {
    int cnt = 0;
    std::string in_line;
    while(getline(test_input, in_line))
    {
      cnt++;
    }
    in_data_size.testing_data_size = cnt;
  }
  test_input.close();

  printf("Size of datasets: traing %d, validation %d, testing %d\n", 
    in_data_size.training_data_size,
    in_data_size.validation_data_size,
    in_data_size.testing_data_size);

  LabeledData* training_labeled_data    = new LabeledData[in_data_size.training_data_size];
  LabeledData* validation_labeled_data  = new LabeledData[in_data_size.validation_data_size];
  LabeledData* testing_labeled_data     = new LabeledData[in_data_size.testing_data_size];
  
  std::cout << "\n---- PREPROC ----\n";
  // Data pre-processing step to create dataset for the classifier
  int output_dim = 
    node_classification_data_preprocessing(
      /* labeled training file name */ training_file_path,
      /* labeled validation file name */ validation_file_path,
      /* labeled testing file name */ testing_file_path,
      /* labeled training data */ training_labeled_data,
      /* labeled validation data */ validation_labeled_data,
      /* labeled testing data */ testing_labeled_data,
      /* data size */ in_data_size
    );

  std::cout << "\n---- CLASSIFIER ----\n";
  std::cout << "output_dim: " << output_dim << std::endl;
  node_classification_classifier(
    /* labeled training data */ training_labeled_data,
    /* labeled validation data */ validation_labeled_data,
    /* labeled testing data */ testing_labeled_data,
    /* size of training/testing datasets */ in_data_size,
    /* node embedding map */ node_emb,
    /* node embedding dimension */ node_embedding_dim,
    /* output dim of the classifier */ output_dim+1,
    /* learning rate */ learning_rate,
    /* number of epochs for training */ num_epochs,
    /* number of neurons in the hidden layer1 */ hidden_layer1_dim,
    /* number of neurons in the hidden layer2 */ hidden_layer2_dim,
    /* batch size */ batch_size,
    /* target validation accuracy */ target_accuracy,
    /* number of threads */ num_threads,
    /* number of workers for parallel data loader */ num_workers
  );

  delete[] training_labeled_data;
  delete[] validation_labeled_data;
  delete[] testing_labeled_data;

  return 0;
}
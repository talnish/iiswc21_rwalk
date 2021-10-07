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
#include "graph.h"
#include "pvector.h"
#include "timer.h"

typedef NodeWeight<NodeID, WeightT> WNode;
typedef EdgePair<NodeID, WNode> Edge;
typedef pvector<Edge> EdgeList;
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
  int   output_dim          =   cli.get_output_dim();
  float learning_rate       =   cli.get_learning_rate();
  int   num_epochs          =   cli.get_num_epochs();
  int   hidden_layer_dim    =   cli.get_hidden_layer_dim();
  int   batch_size          =   cli.get_batch_size();
  float target_accuracy     =   cli.get_target_val_accuracy();

  omp_set_num_threads(48);

  // Compute temporal random walk
  for(int i=0; i<20; ++i) {
    // std::cout << "\n---- RWALK ----\n";
    compute_random_walk(
      /* temporal graph */ g, 
      /* max random walk length */ max_walk_length,
      /* number of rwalks/node */ num_walks_per_node,
      /* filename of random walk */ "out_random_walk.txt"
    );
  }

  return 0;
}
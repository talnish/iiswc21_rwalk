#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <map>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

/*
 * Perform random walk.
 */ 

typedef std::vector<NodeID> NodeVector;
typedef std::vector<std::vector<NodeID>> NodeVectorSet;
typedef std::vector<std::pair<NodeID, WeightT>> TempNodeVector;
typedef std::pair<NodeID, WeightT> TNode;
typedef std::vector<double> DoubleVector;

static std::random_device rd;
static std::mt19937 rng(rd());

/*
  Helper function to print a vector of NodeIDs
*/
void PrintNodeVector(NodeVector in_vec, std::string in_name) {
  std::cout << "Printing " << in_name << "... \n <";
  for(auto it = in_vec.begin(); it < in_vec.end() ; ++it)
    std::cout << *it << ", ";
  std::cout << ">" << std::endl;
}

/*
  Helper function to print a vector of doubles
*/
void PrintDoubleVector(DoubleVector in_vec, std::string in_name) {
  std::cout << "Printing " << in_name << "... \n <";
  for(auto it = in_vec.begin(); it < in_vec.end() ; ++it)
    std::cout << *it << ", ";
  std::cout << ">" << std::endl;
}

/*
  Helper function to sort two temporal edges (TNode)
  in the increasing order of their timestamps
*/
bool SortByTime(TNode n1, TNode n2) {
  return (n1.second < n2.second);
}

/*
  Filter a node's edges that have timestamps larger
  than the incoming edge (i.e., src_time)
*/
TempNodeVector FilterEdgesPostTime(const WGraph &g, NodeID src_node, 
                        WeightT src_time) {
  TempNodeVector filtered_edges;
  for(auto v : g.out_neigh(src_node)) {
    if(v.w >= src_time)
      filtered_edges.push_back(std::make_pair(v, v.w));
  }
  // Sort edges by timestamp?
  //std::sort(filtered_edges.begin(), filtered_edges.end(), SortByTime);
  return filtered_edges;
}

/*
  Random number generator
*/
double RandomNumberGenerator() {
    static std::uniform_real_distribution<double> uid(0,1); 
    return uid(rng);
}

/*
  Function that returns a random neighbor to
  walk based on the probability distribution
*/
int FindNeighborIdx(DoubleVector prob_dist) {
  /* 
  Algorithm: get a random number between 0 and 1
  calculate the CDF of probTimestamps and compare
  against random number
  */
  double curCDF = 0, nextCDF = 0;
  int cnt = 0;
  double random_number = RandomNumberGenerator();
  for(auto it : prob_dist) {
      nextCDF += it;
      if(nextCDF >= random_number && curCDF <= random_number) {
          return cnt;
      }  else {
          curCDF = nextCDF;
          cnt++;
    }
  }
  // Ideal, it should never hit this point
  return rand() % prob_dist.size();
}

/*
  Function to compute the immediate next neighbor to walk
  Builds a probability distribution based on the time difference
  of incoming edge and outgoing edges
*/
bool GetNeighborToWalk(const WGraph &g, NodeID src_node, 
                      WeightT src_time,
                      TNode& next_neighbor) {
  int neighborhood_size = g.out_degree(src_node);
  if(neighborhood_size == 0) {
    return false;
  } else {
    TempNodeVector filtered_edges = FilterEdgesPostTime(g, src_node, src_time);
    if(filtered_edges.empty()) {
      return false;
    }
    if(filtered_edges.size() == 1) {
      next_neighbor = filtered_edges[0];
      return true;
    } else {
      DoubleVector prob_dist;
      WeightT time_boundary_diff;
      time_boundary_diff = g.TimeBoundsDelta(src_node);
      // TODO: parallelism?
      for(auto it : filtered_edges) {
        prob_dist.push_back(exp((float)(it.second-src_time)/time_boundary_diff));
      }
      double exp_sum = std::accumulate(prob_dist.begin(), prob_dist.end(), 0);
      for (uint32_t i = 0; i < prob_dist.size(); ++i) {
        prob_dist[i] = prob_dist[i] / exp_sum;
      }
      int neighbor_index = FindNeighborIdx(prob_dist);
      next_neighbor = filtered_edges[neighbor_index];
      return true;
    }
  }
}

/*
  Function to compute a random walk from a node.
  It recursively calls itself until either the maximum
  walk count has reached or if a node has no latent outgoing edges.
  GetNeighborToWalk() function is used find the immediate 
  next neighbor to walk.
*/
bool ComputeWalkFromANode(const WGraph& g, NodeID src_node,
                          WeightT prev_time_stamp,NodeVector& local_walk, 
                          TNode& next_neighbor_ret, int max_walk_length) {
  TNode next_neighbor;
  if(g.out_degree(src_node) != 0 && 
    GetNeighborToWalk(g, src_node, prev_time_stamp, next_neighbor)) {
    local_walk.push_back(next_neighbor.first);
    next_neighbor_ret = next_neighbor;
    return true;
  }
  return false;
}

/*
  Function to return timestamp to initiate a random walk 
  from a source node.
  Currently it always returns 0, can be modified in the future.
*/
WeightT GetInitialTime(const WGraph &g, NodeID src_node) {
  // TODO: modify this function later
  return (WeightT) 0;
}

/*
  Write random walk to a file
*/
void WriteWalkToAFile(NodeVectorSet global_walk) {
  std::ofstream random_walk_file("out_random_walk.txt");
  for(auto it_out : global_walk) {
    for(auto it_in : it_out) {
      random_walk_file << it_in << " ";
    }
    random_walk_file << "\n";
  }
  random_walk_file.close();
}

/*
  Function that iterates over all vertices in graph,
  and calls ComputeWalkFromANode() function.
  Each random walk from a node is stored in local_walk,
  which is pushed to global_walk that stores all random walks
*/
void ComputeTemporalRandomWalk(const WGraph &g, int max_walk_length) {
  std::cout << "Computing random walk for " << g.num_nodes() << " nodes and " 
    << g.num_edges() << " edges." << std::endl;
  NodeVector local_walk;
  NodeVectorSet global_walk;
  // cilk::reducer< cilk::op_list_append<std::vector<NodeID> > global_walk;
  // mutex m;
  Timer t_rw;
  t_rw.Start();
  // TODO: pragma openmp?
  for(NodeID i = 0; i < g.num_nodes(); ++i) {
    local_walk.push_back(i);
    WeightT prev_time_stamp = 0;
    NodeID next_neighbor = i;
    TNode next_neighbor_ret;
    for(int walk_cnt = 0; walk_cnt < max_walk_length; ++walk_cnt) {
      bool cont = ComputeWalkFromANode(g, next_neighbor, prev_time_stamp,
        local_walk, next_neighbor_ret, max_walk_length);
      if(!cont)
        break;
      next_neighbor = next_neighbor_ret.first;
      prev_time_stamp = next_neighbor_ret.second;
      walk_cnt++;

    }
    global_walk.push_back(local_walk);
    local_walk.clear();
  }
  t_rw.Stop();
  WriteWalkToAFile(global_walk);
  PrintStep("\nTime to compute random walk (s):", t_rw.Seconds());
  printf("\n");
}

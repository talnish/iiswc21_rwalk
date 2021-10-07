// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "reader.h"
#include "writer.h"

using namespace std;

typedef NodeWeight<NodeID, WeightT> WNode;
typedef EdgePair<NodeID, WNode> WEdge;
typedef pvector<WEdge> WEdgeList;
typedef EdgePair<NodeID, NodeID> Edge;
typedef pvector<Edge> EdgeList;


int main(int argc, char* argv[]) {
  CLConvert cli(argc, argv, "converter");
  cli.ParseArgs();
  if (cli.out_weighted()) {
    WEdgeList el;
    WeightedBuilder bw(cli);
    WGraph wg = bw.MakeGraph(&el);
    wg.PrintStats();
    WeightedWriter ww(wg);
    ww.WriteGraph(cli.out_filename(), cli.out_sg());
  } else {
    EdgeList el;
    Builder b(cli);
    Graph g = b.MakeGraph(&el);
    g.PrintStats();
    Writer w(g);
    w.WriteGraph(cli.out_filename(), cli.out_sg());
  }
  return 0;
}

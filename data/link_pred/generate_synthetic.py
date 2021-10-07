import networkx as nx
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodec', required=True)
parser.add_argument('-e', '--edgec', required=True)
parser.add_argument('-s', '--seed', required=True)
args = parser.parse_args()

N=args.nodec
M=args.edgec
S=args.seed
graph = nx.gnm_random_graph(int(N),int(M),int(S))
m = len(graph.edges())
edge2time = {edge: time * 1.0 / m for edge,time in zip(graph.edges(),(m*np.random.rand(m)).astype(int))}

output_filename = "synth_N_" + str(int(N)) + "_E_" + str(int(M)) + "_S_" + str(int(S)) + "_preproc.wel"
print("Output preproc wel to ", output_filename)
out_file = open(output_filename, "w")
for (k, v) in edge2time.items():
    out_file.write(str(k[0]) + ' ' + str(k[1]) + ' ' + str(v) + '\n')
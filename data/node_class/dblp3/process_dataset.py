import multiprocessing
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from multiprocessing import Pool
import argparse
import os
from collections import defaultdict


def write_to_label(graph_l, num_of_nodes, data_label):
    dest_label = {}
    for i in range(num_of_nodes):
        # assert(len(np.where(data['labels'][i])) == 1)
        dest_label[i] = np.where(data_label[i] == 1)[0][0]
        # print(dest_label[i])
    out_label_f = open(graph_l, "w+")
    for i in range(len(dest_label)):
        out_label_f.write(str(i) + ' ' + str(dest_label[i]) + "\n")
    out_label_f.close()

def write_to_file(graph_f, data_adj, file_idx, seeds):
    max_connected_node_id = 0
    print("Writing to ", graph_f + str(file_idx))
    with open(graph_f + str(file_idx), 'w+') as f:
        for seed in tqdm(seeds):
            for j in range(num_of_timestamps):
                # print("j:", j)
                for dest in np.where(data_adj[j][seed] == 1)[0]:
                    # out_f.write(str(i) + ' ' + str(dest) + ' ' + str(dest_label[i]) + ' ' + str(j) + '\n')
                    f.write(str(seed) + ' ' + str(dest) + ' ' + str(j * 1.0 / num_of_timestamps) + '\n')
                    # return
                    max_connected_node_id = seed
    return max_connected_node_id

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True)
    parser.add_argument('--thread', type=int, default=8)
    args = parser.parse_args()
    
    partition = args.thread

    # dataset_name='Brain'
    data = np.load(args.i)
    num_of_timestamps = data['adjs'].shape[0]
    num_of_nodes = data['adjs'].shape[1]
    num_of_labels = data['labels'].shape[1]
    graph_l = args.i + "_labels.out"
    write_to_label(graph_l, num_of_nodes, data['labels'])
    
    # graph_f = args.i + "_graph.out"
    graph_f = "tgraph.wel"
    max_connected_node_id = write_to_file(graph_f, data['adjs'], "", list(range(0, num_of_nodes)))

    output_train_filename = "train.tsv"
    out_train_file = open(output_train_filename, "w")
    output_valid_filename = "valid.tsv"
    out_valid_file = open(output_valid_filename, "w")
    output_test_filename = "test.tsv"
    out_test_file = open(output_test_filename, "w")

    temp_edge_list = np.genfromtxt(args.i + "_labels.out", delimiter=' ')
    os.remove(args.i + "_labels.out")
    
    sorted_edge_list = np.array(sorted(temp_edge_list, key=lambda x:x[0]))
    sorted_edge_list = sorted_edge_list[sorted_edge_list[:,0] <= max_connected_node_id]

    limit = 300
    d = defaultdict(lambda: 0, {})
    arr = np.zeros((0,2),int)

    for i in range(len(sorted_edge_list)):
        if d[sorted_edge_list[i][1]] <= limit:
            arr = np.append(arr, [[int(sorted_edge_list[i][0]), int(sorted_edge_list[i][1])]], axis=0)
            d[sorted_edge_list[i][1]] += 1
    sorted_edge_list = arr

    indices = np.random.permutation(len(sorted_edge_list))
    # ratios: 60% for training, 20% for validation, 20% for testing
    # feel free to change them as you find fit
    train_size = int(len(sorted_edge_list) * 0.6)
    valid_size = int(len(sorted_edge_list) * 0.2)
    test_size = int(len(sorted_edge_list) * 0.2)

    training_idx, valid_idx, test_idx = indices[:train_size], indices[train_size:train_size+valid_size], indices[train_size+valid_size:]

    for i in training_idx:
        out_train_file.write(str(int(sorted_edge_list[i][0])))
        out_train_file.write(" ")
        out_train_file.write(str(int(sorted_edge_list[i][1])))
        out_train_file.write("\n")
    
    for i in test_idx:
        out_test_file.write(str(int(sorted_edge_list[i][0])))
        out_test_file.write(" ")
        out_test_file.write(str(int(sorted_edge_list[i][1])))
        out_test_file.write("\n")

    for i in valid_idx:
        out_valid_file.write(str(int(sorted_edge_list[i][0])))
        out_valid_file.write(" ")
        out_valid_file.write(str(int(sorted_edge_list[i][1])))
        out_valid_file.write("\n")

    print("Wrote to the following files:")
    print(output_train_filename)
    print(output_valid_filename)
    print(output_test_filename)

    out_test_file.close()
    out_valid_file.close()
    out_train_file.close()
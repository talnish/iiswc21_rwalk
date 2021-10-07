/*
 * Custom data loader module for link prediction.
 */

template<typename T>
void print_vector_of_pairs(std::vector<std::pair<T,T>> in_vec) 
{
    for(auto it : in_vec)
        std::cout << "{" << it.first << "," << it.second << "} ";
    std::cout << "\n";
}

template<typename T>
void print_vector(std::vector<T> in_vec) 
{
    for(auto it : in_vec)
        std::cout << it << " ";
    std::cout << "\n";
}

template<typename T>
void print_set(std::set<T> in_vec) 
{
    for(auto it : in_vec)
        std::cout << it << "\n";
}

template<typename T1, typename T2>
void print_map(std::map<T1, std::vector<T2>> in_map)
{
    for(auto it : in_map)
    {
        std::cout << "key: " << it.first << ", value: ";
        print_vector(it.second);
        std::cout << std::endl;
    }
}

void print_tensor_array(torch::Tensor* t_array, int len)
{
    for(int i=0; i<len; ++i)
        std::cout << "*** [" << i << "]\n" << t_array[i] << std::endl;
}

void compute_edge_features_labels(
    EdgePairStruct* edge_list,
    NodeEmb node_emb,
    int node_embedding_dim,
    long long int train_test_dataset_size,
    std::vector<torch::Tensor>* edge_features,
    std::vector<torch::Tensor>* edge_labels,
    float label_val
)
{
    int edge_emb_size = 2 * node_embedding_dim;
    for(long long int i=0; i<train_test_dataset_size; ++i)
    {
        std::vector<float> src_emb, dst_emb;
        src_emb =  node_emb.at(edge_list[i].src_node);
        dst_emb =  node_emb.at(edge_list[i].dst_node);
        // Concatenate both source and destination embeddings...
        src_emb.insert(src_emb.end(), dst_emb.begin(), dst_emb.end());
        torch::Tensor this_edge_emb = 
            torch::from_blob(src_emb.data(), {edge_emb_size}).clone();
        torch::Tensor this_edge_label = 
            torch::full({1}, label_val); //, torch::kLong);
        edge_features->push_back(this_edge_emb);
        edge_labels->push_back(this_edge_label);
        // std::cout << "src: " << edge_list[i].src_node << ", dst: " << edge_list[i].dst_node << std::endl;
        // std::cout << "edge features: " << this_edge_emb << std::endl;
    }
}

void compute_edge_features_labels_opt(
    EdgePairStruct* edge_list,
    NodeEmb node_emb,
    int node_embedding_dim,
    long long int train_test_dataset_size,
    torch::Tensor* edge_features,
    torch::Tensor* edge_labels,
    float label_val
)
{
    int edge_emb_size = 2 * node_embedding_dim;
    for(long long int i=0; i<train_test_dataset_size; ++i)
    {
        long long int idx = i;
        if(label_val == 0)
            idx = train_test_dataset_size + i;
        std::vector<float> src_emb, dst_emb;
        src_emb =  node_emb.at(edge_list[i].src_node);
        dst_emb =  node_emb.at(edge_list[i].dst_node);
        // Concatenate both source and destination embeddings...
        src_emb.insert(src_emb.end(), dst_emb.begin(), dst_emb.end());
        torch::Tensor this_edge_emb = 
            torch::from_blob(src_emb.data(), {edge_emb_size}/*, dtype(torch::kFloat16)*/).clone()/*.to(torch::kCUDA)*/;    // Uncomment to try float 16, or transfer the dataset to GPU before training features
        torch::Tensor this_edge_label = 
            torch::full({1}, label_val)/*.clone()*//*.to(torch::kCUDA)*/; //, torch::kLong);
        edge_features[idx] = this_edge_emb;
        edge_labels[idx] = this_edge_label;
    }
}

class CustomDataset : public torch::data::Dataset<CustomDataset> 
{
private:
    EdgePairStruct* p_list;
    EdgePairStruct* n_list;
    NodeEmb node_emb;
    int node_embedding_dim;
    long long int train_test_dataset_size;
    torch::Tensor* edge_features_priv;
    torch::Tensor* edge_labels_priv;
    CustomDataset * ptr;
public:
    ~CustomDataset(){
        // printf("Destructor of %#010x is called\n", ptr);
    }

    void clean_pn(){
        delete[] p_list;
        delete[] n_list;
    }

    void clean_edge(){
        delete[] edge_features_priv;
        delete[] edge_labels_priv;
    }

    // Constructor
    CustomDataset(
        const WGraph &g,
        EdgePairStruct* p_list_in,
        EdgePairStruct* n_list_in,
        NodeEmb node_emb_in,
        int node_embedding_dim_in,
        long long int train_test_dataset_size_in
    )
    {
        ptr = this;
        node_emb = node_emb_in;
        node_embedding_dim = node_embedding_dim_in;
        train_test_dataset_size = train_test_dataset_size_in;

        // p_list   = new EdgePairStruct[train_test_dataset_size];
        // n_list   = new EdgePairStruct[train_test_dataset_size];
        p_list   = p_list_in;
        n_list   = n_list_in;
        
        edge_features_priv = new torch::Tensor[2 * train_test_dataset_size];
        edge_labels_priv   = new torch::Tensor[2 * train_test_dataset_size];
        std::cout << "Computing p_list edge features\n";
        compute_edge_features_labels_opt(
            p_list, 
            node_emb, 
            node_embedding_dim,
            train_test_dataset_size,
            edge_features_priv, 
            edge_labels_priv,
            1
        );
        
        std::cout << "Computing n_list edge features\n";
        compute_edge_features_labels_opt(
            n_list,
            node_emb, 
            node_embedding_dim,
            train_test_dataset_size,
            edge_features_priv, 
            edge_labels_priv,
            0
        );

        // Print datasets for debugging?
        // CAUTION: This will print the entire training/testing datasets
        //          Can fill up the terminal and slow down the program!
        if(print_datasets)
            print_data_elements();

        clean_pn();
    };

    void print_data_elements()
    {
        std::cout << "p_list...\n";
        print_edge_pair(p_list, train_test_dataset_size);
        std::cout << "n_list...\n";
        print_edge_pair(n_list, train_test_dataset_size);
        std::cout << "^^^^^^^^^^^\n";
        print_tensor_array(edge_features_priv, 2 * train_test_dataset_size);
        print_tensor_array(edge_labels_priv, 2 * train_test_dataset_size);
    };

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override
    {
        torch::Tensor sample_node_id = edge_features_priv[index];
        torch::Tensor sample_label   = edge_labels_priv[index];
        return {sample_node_id.clone(), sample_label.clone()};
    };

    // Return the length of data
    torch::optional<size_t> size() const override
    {
        // train_test_dataset_size is the size of each positive/negative sample sets
        // total dataset size is 2 * train_test_dataset_size because it has both
        //     positive and negative samples
        return 2 * train_test_dataset_size;
    };
};
/*
 * Custom data loader module for node classification.
 */

template<typename T>
void print_vector_of_pairs(std::vector<std::pair<T,T>> in_vec) 
{
    for(auto it : in_vec)
        std::cout << it.first << "," << it.second << "\n";
}

template<typename T>
void 
print_vector(std::vector<T> in_vec) 
{
    for(auto it : in_vec)
        std::cout << it << " ";
}

template<typename T>
void 
print_set(std::set<T> in_vec) 
{
    for(auto it : in_vec)
        std::cout << it << "\n";
}

template<typename T1, typename T2>
void 
print_map(std::map<T1, std::vector<T2>> in_map)
{
    for(auto it : in_map)
    {
        std::cout << "key: \n" << it.first << "\nvalue: \n";
        print_vector(it.second);
        std::cout << "---\n";
    }
}

void prepare_data_opt(
    LabeledData* labeled_data,
    long long int data_size,
    NodeEmb node_emb,
    int node_emb_dim,
    torch::Tensor* node_features,
    torch::Tensor* node_labels
)
{
    for(long long int i=0; i<data_size; ++i)
    {
        std::vector<float> this_node_emb = node_emb.at(labeled_data[i].node_id);
        torch::Tensor this_emb = torch::from_blob(this_node_emb.data(), 
            {node_emb_dim}).clone();
        torch::Tensor this_label = torch::full({1}, labeled_data[i].node_label, torch::kLong);
        node_features[i] = this_emb;
        node_labels[i] = this_label;
    }
}

class CustomDataset : public torch::data::Dataset<CustomDataset> 
{
private:
    LabeledData* labeled_data;
    long long int data_size;
    int node_emb_dim;
    NodeEmb node_emb;
    torch::Tensor* node_emb_priv;
    torch::Tensor* labels_priv;
public:
    // Constructor
    CustomDataset(
        LabeledData* labeled_data_in,
        long long int data_size_in,
        NodeEmb node_emb_in,
        int node_emb_dim_in) 
    {
        data_size = data_size_in;
        node_emb_dim = node_emb_dim_in;
        labeled_data = new LabeledData[data_size];
        labeled_data = labeled_data_in;
        node_emb = node_emb_in;

        node_emb_priv = new torch::Tensor[data_size];
        labels_priv = new torch::Tensor[data_size];

        prepare_data_opt(
            labeled_data,
            data_size,
            node_emb,
            node_emb_dim,
            node_emb_priv, 
            labels_priv
        );
        
        if(print_datasets)
        {
            for(long long int i=0; i<data_size; ++i)
            {
                std::cout << "node: \n" << node_emb_priv[i] 
                    << ", id: " << labels_priv[i] << std::endl;
            }
        }
    };

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override
    {
        // std::cout << "index: " << index << std::endl;
        torch::Tensor sample_node_id = node_emb_priv[index];
        torch::Tensor sample_label   = labels_priv[index];
        return {sample_node_id.clone(), sample_label.clone()};
    };

    // Return the length of data
    torch::optional<size_t> size() const override
    {
        return data_size;
    };

};
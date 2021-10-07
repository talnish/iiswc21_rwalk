/*
 * Optimized data pre-processing for link prediction.
 * Applied optimized data structures and parallelism in negative_sampling().
 */

void corrupt_tail(
    const WGraph &g, 
    EdgePairStruct* n_list,
    EdgePairStruct edge_in,
    long long int i,
    int64_t num_nodes)
{
    while(1)
    {
        static std::uniform_int_distribution<NodeID> uid(0,num_nodes-1);
        NodeID new_dst = uid(rng);
        if(!g.EdgeExists(edge_in.src_node, new_dst))
        {
            if((new_dst > num_nodes-1) || (new_dst < 0)) {
                std::cout << "wrong node_id 2\n";
            }
            n_list[i].src_node = edge_in.src_node;
            n_list[i].dst_node = new_dst;
            break;
        }
    }
}

void corrupt_both(
    const WGraph &g, 
    EdgePairStruct* n_list,
    EdgePairStruct edge_in,
    long long int i,
    int64_t num_nodes)
{
    while(1)
    {
        static std::uniform_int_distribution<NodeID> uid1(0,num_nodes-1);
        static std::uniform_int_distribution<NodeID> uid2(0,num_nodes-1);
        NodeID new_src = uid1(rng);
        NodeID new_dst = uid2(rng);
        if(!g.EdgeExists(new_src, new_dst))
        {
            if((new_src < 0) || (new_dst < 0) 
                || (new_src > num_nodes-1) || (new_dst > num_nodes-1)) {
                std::cout << "wrong node_id 1/2\n";
            }
            n_list[i].src_node = new_src;
            n_list[i].dst_node = new_dst;
            break;
        }
    }
}

void negative_sampling(
    const WGraph &g,
    EdgePairStruct* p_list,
    EdgePairStruct* n_list,
    long long int test_train_data_size,
    int64_t num_nodes)
{
    // EdgePairStruct* n_list = new EdgePairStruct[test_train_data_size];
    parallel_for(long long int i=0; i<test_train_data_size; ++i)
    {
        static std::uniform_real_distribution<double> uid(0,1); 
        if(uid(rng) > 0.5)
        {
            corrupt_tail(g, n_list, p_list[i], i, g.num_nodes());
        } else
        {
            corrupt_both(g, n_list, p_list[i], i, g.num_nodes());
        }
    }
}

void sanitize_data(
    EdgePairStruct* pn_list,
    int64_t num_nodes,
    long long int test_train_data_size
)
{
    for(long long int i=0; i<test_train_data_size; ++i)
    {
        if(pn_list[i].src_node < 0 || pn_list[i].src_node > num_nodes-1) {
            std::cout << "*** Wrong node id: " << pn_list[i].src_node << ", index: " << i << std::endl;
            exit(1);
        }
        if(pn_list[i].dst_node < 0 || pn_list[i].dst_node > num_nodes-1) {
            std::cout << "*** Wrong node id: " << pn_list[i].dst_node << ", index: " << i << std::endl;
            exit(1);
        }
    }
}

void print_temp_el(TempELStruct* temp_el, int edge_cnt)
{
    for(int i=0; i<edge_cnt; ++i)
    {
        printf("{%ld, %ld, %f}\n", temp_el[i].src_node, temp_el[i].dst_node, temp_el[i].time_stamp);
    }
}

void print_edge_pair(EdgePairStruct* edge_pair, int edge_cnt)
{
    for(int i=0; i<edge_cnt; ++i)
    {
        printf("{%ld, %ld}\n", edge_pair[i].src_node, edge_pair[i].dst_node);
    }
}

bool SortStructByTime(TempELStruct tmp1, TempELStruct tmp2) 
{
  return (tmp1.time_stamp < tmp2.time_stamp);
}


void link_prediction_data_preprocessing
(
    const WGraph &g, 
    EdgeList &el,
    TempELStruct* temp_el,
    EdgePairStruct* train_p_list,
    EdgePairStruct* train_n_list,
    EdgePairStruct* test_p_list,
    EdgePairStruct* test_n_list,
    EdgePairStruct* valid_p_list,
    EdgePairStruct* valid_n_list,
    // float ratio
    long long int train_dataset_size,
    long long int test_dataset_size,
    long long int valid_dataset_size
)
{
    std::cout << "Preprocessing data...\n";

    Timer t_data_preproc;
    t_data_preproc.Start();
    long long int edge_cnt = 0;
    for(size_t e=0; e<el.size(); ++e)
    {
        temp_el[edge_cnt].time_stamp = el[e].v.w;
        temp_el[edge_cnt].src_node   = el[e].u;
        temp_el[edge_cnt].dst_node   = el[e].v.v;
        edge_cnt++;
    }
    
    // Sort the edge list according to time stamps
    std::sort(temp_el, (temp_el+(g.num_edges())), SortStructByTime);

    // Separate training and testing edge sets
    // long long int test_train_data_size = g.num_edges() * (1 - ratio);
    EdgePairStruct* potential_train_p_list = new EdgePairStruct[edge_cnt - test_dataset_size];
    long long int train_cnt = 0, test_cnt = 0;
    for(long long int i=0; i<edge_cnt; ++i)
    {
        if(i < (edge_cnt - test_dataset_size))
        {
            potential_train_p_list[train_cnt].src_node = temp_el[i].src_node;
            potential_train_p_list[train_cnt].dst_node = temp_el[i].dst_node;
            train_cnt++;
        } else
        {
            test_p_list[test_cnt].src_node = temp_el[i].src_node;
            test_p_list[test_cnt].dst_node = temp_el[i].dst_node;
            test_cnt++;
        }
    }

    // Sample training set
    std::experimental::sample
    (
        potential_train_p_list, 
        potential_train_p_list + (edge_cnt - test_dataset_size),
        train_p_list,
        train_dataset_size,
        std::mt19937{std::random_device{}()}
    );

    // Sample validation set
    std::experimental::sample
    (
        potential_train_p_list,
        potential_train_p_list + (edge_cnt - test_dataset_size),
        valid_p_list,
        valid_dataset_size,
        std::mt19937{std::random_device{}()}
    );

    delete[] potential_train_p_list;

    std::cout << "Negative sampling...\n";
    Timer t_neg_sampl;
    t_neg_sampl.Start();
    
    negative_sampling(g, train_p_list, train_n_list, train_dataset_size, g.num_nodes());
    negative_sampling(g, valid_p_list, valid_n_list, valid_dataset_size, g.num_nodes());
    negative_sampling(g, test_p_list,  test_n_list,  test_dataset_size,  g.num_nodes());
    
    t_neg_sampl.Stop();
    t_data_preproc.Stop();

    PrintStep("[TimingStat] Negative sampling time     (s):", t_neg_sampl.Seconds());
    PrintStep("[TimingStat] Data pre-preprocssing time (s):", t_data_preproc.Seconds());

    // Print datasets for debugging?
    // CAUTION: This will print the entire training/testing datasets
    //          Can fill up your terminal and slow down the program!
    if(print_datasets)
    {
        std::cout << "train_p_list...\n";
        print_edge_pair(train_p_list, train_dataset_size);
        std::cout << "train_n_list...\n";
        print_edge_pair(train_n_list, train_dataset_size);
        std::cout << "test_p_list...\n";
        print_edge_pair(test_p_list, test_dataset_size);
        std::cout << "test_n_list...\n";
        print_edge_pair(test_n_list, test_dataset_size);
    }
    
}
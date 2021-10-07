/*
 * Data pre-processing for link prediction.
 */

NodePair corrupt_tail(
    const WGraph &g, 
    NodePair edge_in,
    ELPair edge_list_connections, 
    int64_t num_nodes)
{
    NodePair new_edge;
    while(1)
    {
        static std::uniform_int_distribution<NodeID> uid(0,num_nodes-1);
        NodeID new_dst = uid(rng);
        if(!g.EdgeExists(edge_in.first, new_dst))
        {
            new_edge = std::make_pair(edge_in.first, new_dst);
            break;
        }
        // new_edge = std::make_pair(edge_in.first, new_dst);
        // if(std::find(edge_list_connections.begin(), 
        //     edge_list_connections.end(), 
        //     new_edge) == edge_list_connections.end())
        // {
        //     // Created an edge that does not exist
        //     break;
        // }
    }
    return new_edge;
}

NodePair corrupt_both(
    const WGraph &g, 
    NodePair edge_in,
    ELPair edge_list_connections, 
    int64_t num_nodes)
{
    NodePair new_edge;
    while(1)
    {
        static std::uniform_int_distribution<NodeID> uid1(0,num_nodes-1);
        static std::uniform_int_distribution<NodeID> uid2(0,num_nodes-1);
        NodeID new_src = uid1(rng);
        NodeID new_dst = uid2(rng);
        if(!g.EdgeExists(new_src, new_dst))
        {
            new_edge = std::make_pair(new_src, new_dst);
            break;
        }
        // new_edge = std::make_pair(new_src, new_dst);
        // if(std::find(edge_list_connections.begin(), 
        //     edge_list_connections.end(), 
        //     new_edge) == edge_list_connections.end())
        // {
        //     // Created an edge that does not exist
        //     break;
        // }
    }

    return new_edge;   
}

ELPair negative_sampling(
    const WGraph &g, 
    ELPair train_list,
    ELPair edge_list_connections, 
    int64_t num_nodes)
{
    ELPair test_list;
    for(auto it : train_list)
    {   
        NodePair new_edge;
        static std::uniform_real_distribution<double> uid(0,1); 
        if(uid(rng) > 0.5)
        {
            new_edge = corrupt_tail(g, it, edge_list_connections, num_nodes);
        } else
        {
            new_edge = corrupt_both(g, it, edge_list_connections, num_nodes);
        }
        test_list.push_back(new_edge);
    }
    return test_list;
}

bool incr_time(EdgeTuple e1, EdgeTuple e2)
{
    return (std::get<2>(e1) < std::get<2>(e2));
}

void link_prediction_data_preprocessing(
    const WGraph &g, 
    EdgeList &el,
    ELPair &train_p_list,
    ELPair &train_n_list,
    ELPair &test_p_list,
    ELPair &test_n_list,
    float ratio)
{
    // Intermediate ELPair
    ELPair int_train_p_list;
    ELPair int_train_n_list;
    ELPair int_test_p_list;
    ELPair int_test_n_list;

    std::cout << "Preprocessing data...\n";
    TempEL edge_list_time;
    ELPair edge_list_connections, edge_list_subgraph;
    NodeSet node_set;

    Timer t_data_preproc;
    t_data_preproc.Start();
    for(size_t e=0; e<el.size(); ++e)
    {
        edge_list_time.push_back(std::make_tuple(el[e].u, el[e].v.v, el[e].v.w));
        edge_list_connections.push_back(std::make_pair(el[e].u, el[e].v.v));
        node_set.insert(el[e].u);
        node_set.insert(el[e].v.v);
    }

    // Sort the edge_list_time in increasing order
    std::sort(edge_list_time.begin(), edge_list_time.end(), incr_time);

    int cnt=0;
    for(auto it : edge_list_time)
    {
        if(cnt < ratio * edge_list_time.size())
        {
            edge_list_subgraph.push_back(
                std::make_pair(std::get<0>(it), std::get<1>(it)));
        } else 
        {
            int_test_p_list.push_back(
                std::make_pair(std::get<0>(it), std::get<1>(it)));
        }
        cnt++;
    }

    std::experimental::sample(edge_list_subgraph.begin(), edge_list_subgraph.end(), 
        std::back_inserter(int_train_p_list),
        int_test_p_list.size(), std::mt19937{std::random_device{}()});

    std::cout << "Negative sampling...\n";
    Timer t_neg_sampl;
    t_neg_sampl.Start();
    int_train_n_list = negative_sampling(g, int_train_p_list, edge_list_connections, g.num_nodes());
    int_test_n_list  = negative_sampling(g, int_test_p_list, edge_list_connections, g.num_nodes());
    t_neg_sampl.Stop();
    t_data_preproc.Stop();
    PrintStep("\nTime to perform negative sampling (s):", t_neg_sampl.Seconds());
    PrintStep("\nTime for pre-processing data (s)     :", t_data_preproc.Seconds());

    // Filter out the redundant edges
    for(auto it : int_train_p_list)
    {
        if(std::find(train_p_list.begin(), train_p_list.end(), it) == train_p_list.end())
        {
            train_p_list.push_back(it);
        }
    }
    for(auto it : int_train_n_list)
    {
        if(std::find(train_n_list.begin(), train_n_list.end(), it) == train_n_list.end())
        {
            train_n_list.push_back(it);
        }
    }
    for(auto it : int_test_p_list)
    {
        if(std::find(test_p_list.begin(), test_p_list.end(), it) == test_p_list.end())
        {
            test_p_list.push_back(it);
        }
    }
    for(auto it : int_test_n_list)
    {
        if(std::find(test_n_list.begin(), test_n_list.end(), it) == test_n_list.end())
        {
            test_n_list.push_back(it);
        }
    }

    // Print to a file
    std::ofstream train_p_file("train_p.tsv");
    std::ofstream train_n_file("train_n.tsv");
    std::ofstream test_p_file("test_p.tsv");
    std::ofstream test_n_file("test_n.tsv");

    for(auto it : train_p_list)
    {
        train_p_file << it.first << "\t" << it.second << "\n";
    }
    for(auto it : train_n_list)
    {
        train_n_file << it.first << "\t" << it.second << "\n";
    }
    for(auto it : test_p_list)
    {
        test_p_file << it.first << "\t" << it.second << "\n";
    }
    for(auto it : test_n_list)
    {
        test_n_file << it.first << "\t" << it.second << "\n";
    }

    train_p_file.close();
    train_n_file.close();
    test_p_file.close();
    test_n_file.close();

}
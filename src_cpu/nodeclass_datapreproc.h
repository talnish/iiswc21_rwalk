/*
 * Data pre-processing for node classification.
 * Reads labeled node data from training/testing files.
 */

int
read_labeled_data(
    std::string input_file_path,
    LabeledData* labeled_data) 
{
    std::set<int> label_categories;
    int num_categories;
    LabeledNode labeled_nodes_from_file;
    std::ifstream f_input(input_file_path);
    if(f_input)
    {
        std::string in_line;
        int cnt = 0;
        while(getline(f_input, in_line))
        {
            std::istringstream buf(in_line);
            std::istream_iterator<std::string> start(buf), end;
            std::vector<std::string> tokens(start, end);
            labeled_data[cnt].node_id = stoi(tokens[0]);
            labeled_data[cnt].node_label = stoi(tokens[1]);
            label_categories.insert(stoi(tokens[1]));
            cnt++;
        }
    }
    f_input.close();
    
    // return the number of label categories
    // one-hot encoding in the classifier expects values starting from 0
    // hence, if the value 0 is not part of the label categories, add a dummy output node
    // otherwise the training loop will break because of inconsistent output size
    if(label_categories.find(0) != label_categories.end()) num_categories = label_categories.size();
    else num_categories = label_categories.size() + 1;
    return num_categories;
}

int 
node_classification_data_preprocessing(
    std::string training_file,
    std::string validation_file,
    std::string testing_file,
    LabeledData* training_labeled_data,
    LabeledData* validation_labeled_data,
    LabeledData* testing_labeled_data,
    InputDataSize in_data_size)
{
    Timer t_dataread;
    std::cout << "Reading labeled data...\n";
    t_dataread.Start();
    int training_label_categories = 0, 
        validation_label_categories = 0, 
        testing_label_categories = 0;
    training_label_categories  = read_labeled_data(training_file, training_labeled_data);
    if(in_data_size.validation_data_size > 0)
        validation_label_categories = read_labeled_data(validation_file, validation_labeled_data);
    testing_label_categories    = read_labeled_data(testing_file, testing_labeled_data);
    t_dataread.Stop();
    PrintStep("[TimingStat] Data-read time (s):", t_dataread.Seconds());

    std::cout << "Number of label categories in training dataset    : " << training_label_categories << std::endl;
    if(in_data_size.validation_data_size > 0)
        std::cout << "Number of label categories in validation dataset  : " << validation_label_categories << std::endl;
    std::cout << "Number of label categories in testing dataset     : " << testing_label_categories << std::endl;

    return training_label_categories;
}
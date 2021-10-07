/*
 * Classifier module for link prediction that does both training and inference.
 */ 

void link_prediction_clasifier(
    const WGraph &g, 
    EdgePairStruct* train_p_list, 
    EdgePairStruct* train_n_list, 
    EdgePairStruct* test_p_list,
    EdgePairStruct* test_n_list,
    EdgePairStruct* valid_p_list,
    EdgePairStruct* valid_n_list,
    NodeEmb node_emb,
    int node_embedding_dimension,
    long long int train_dataset_size,
    long long int test_dataset_size,
    long long int valid_dataset_size,
    // hyper parameters
    int output_size,
    float learning_rate,
    int num_epochs,
    int hidden_layer_dimension,
    int batch_size,
    float target_val_accuracy,
    int num_threads,
    int num_workers)
{
    int input_size  = 2 * node_embedding_dimension;

    // Custom data loader for training set
    std::cout << "Loading training dataset...\n";
    auto training_custom_ptr = new CustomDataset(
        g,
        train_p_list,
        train_n_list,
        node_emb,
        node_embedding_dimension,
        train_dataset_size);
    auto training_custom_dataset = (*training_custom_ptr).map(torch::data::transforms::Stack<>());
    // const size_t training_batch_size = training_custom_dataset.size().value() / num_batches;
    // Sampler types: SequentialSampler, RandomSampler
    auto train_data_loader = 
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(training_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    // Custom data loader for validation set
    std::cout << "Loading validation dataset...\n";
    auto validation_custom_ptr = new CustomDataset(
        g,
        valid_p_list,
        valid_n_list,
        node_emb,
        node_embedding_dimension,
        valid_dataset_size);
    auto validation_custom_dataset = (*validation_custom_ptr).map(torch::data::transforms::Stack<>());
    const size_t validation_data_size = validation_custom_dataset.size().value();
    // const size_t validation_batch_size = validation_custom_dataset.size().value() / num_batches;
    // Sampler types: SequentialSampler, RandomSampler
    auto valid_data_loader = 
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(validation_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    // Create a network
    auto net = std::make_shared<Net>
    (
        input_size,
        output_size,
        hidden_layer_dimension
    );

    assert(torch::cuda::is_available());    // This branch is supposed to run on GPU only
                                            // We do not want torch training triggered on CPU
    torch::Device device = torch::kCUDA;
    std::cout << "CUDA available! Training on GPU." << std::endl;
    net->to(device);

    // Optimizer
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    /* Training phase */
    std::cout << "Training...\n";
    // net->train();

    Timer t_training;
    t_training.Start();

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        int batch_index = 0;
        float loss_acc = 0;
        // Iterate the data loader to yield batches from the dataset.
        // Testing loop
        net->train();
        for (auto& batch : *train_data_loader) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            auto prediction = net->forward(batch.data.to(torch::kCUDA));
            // Compute a loss value to judge the prediction of our model.
            auto loss = torch::binary_cross_entropy(prediction, batch.target.to(torch::kCUDA));
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            loss_acc += loss.item<float>();
            batch_index++;
        }
        std::cout << "[Training] Epoch-number: " << epoch;
        std::cout << "; training-loss: " << loss_acc / batch_index << std::endl;
        if(epoch % 10 == 0) {
            net->eval();
            int val_correct_samples = 0;
            for (const auto& batch : *valid_data_loader) {
                auto data = batch.data.to(torch::kCUDA);
                auto targets = batch.target.to(torch::kCUDA).view({-1});
                auto output = net->forward(data);
                auto loss = torch::binary_cross_entropy(output, targets);
                
                val_correct_samples += torch::sum(torch::eq(torch::round(output.view({-1})), targets)).item<int>(); 
            }
            float val_accuracy = (float) val_correct_samples / validation_data_size;
            std::cout << "[Validation]     validation-accuracy: " << val_accuracy << std::endl;
            if(val_accuracy >= target_val_accuracy) {
                std::cout << "Target validation accuracy of " << target_val_accuracy << " reached!\n" 
                        << "Breaking from training loop at " << epoch << " epochs...\n";
            break;
            }
        }
    }

    t_training.Stop();
    PrintStep("[TimingStat] Training time (s):", t_training.Seconds());

    std::cout << "Training finished!\n";

    training_custom_ptr->clean_edge();
    validation_custom_ptr->clean_edge();

    /* Testing phase */
    // Custom data loader for testing set
    std::cout << "Loading testing dataset...\n";
    auto testing_custom_ptr = new CustomDataset(
        g,
        test_p_list,
        test_n_list,
        node_emb,
        node_embedding_dimension,
        test_dataset_size);
    auto testing_custom_dataset = (*testing_custom_ptr).map(torch::data::transforms::Stack<>());
    const size_t testing_data_size = testing_custom_dataset.size().value();
    // Sampler types: SequentialSampler, RandomSampler
    auto test_data_loader = 
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(testing_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    std::cout << "Testing...\n";
    net->eval();

    int num_correct = 0;

    Timer t_testing;
    t_testing.Start();

    for (const auto& batch : *test_data_loader) {
        auto data = batch.data.to(torch::kCUDA);
        auto targets = batch.target.to(torch::kCUDA).view({-1});

        auto output = net->forward(data);
        
        auto loss = torch::binary_cross_entropy(output, targets);
        
        num_correct += torch::sum(torch::eq(torch::round(output.view({-1})), targets)).item<int>();
    }
    
    t_testing.Stop();
    PrintStep("[TimingStat] Testing  time (s):", t_testing.Seconds());

    std::cout << "Total correct predictions: " << num_correct << std::endl;

    std::cout << "Testing accuracy: " << (float)num_correct/testing_data_size << std::endl;

    testing_custom_ptr->clean_edge();
}
/*
 * Classifier module for node classification that does both training and inference.
 */ 

void node_classification_classifier(
    LabeledData* training_labeled_data,
    LabeledData* validation_labeled_data,
    LabeledData* testing_labeled_data,
    InputDataSize in_data_size,
    NodeEmb node_emb,
    int node_embedding_dim,
    // Hyperparameters
    int output_size,
    float learning_rate,
    int num_epochs,
    int hidden_layer1_dimension,
    int hidden_layer2_dimension,
    int batch_size,
    float target_val_accuracy,
    int num_threads,
    int num_workers)
{
    int input_size = node_embedding_dim;

    // num_workers is input to custom data loader
    // num_workers = 0 is default to use the main thread 
    // For large num_threads, num_workers = 8 was found to be optimal.
    // https://pytorch.org/tutorials/advanced/cpp_frontend.html
    // int num_workers = 4;
    // if(num_threads < 8) num_workers = 4;
    // else num_workers = 4;

    // Custom data loader for training set
    std::cout << "Loading training dataset...\n";
    auto training_custom_dataset = CustomDataset(
        training_labeled_data,
        in_data_size.training_data_size,
        node_emb,
        node_embedding_dim).map(torch::data::transforms::Stack<>());
    // int training_batch_size = training_custom_dataset.size().value() / num_batches;
    // Sampler types: SequentialSampler, RandomSampler
    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(training_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    // Custom data loader for validation set
    std::cout << "Loading validation dataset...\n";
    auto validation_custom_dataset = CustomDataset(
        validation_labeled_data,
        in_data_size.validation_data_size,
        node_emb,
        node_embedding_dim).map(torch::data::transforms::Stack<>());
    // int validation_batch_size = validation_custom_dataset.size().value() / num_batches;
    const size_t validation_data_size = validation_custom_dataset.size().value();
    // Sampler types: SequentialSampler, RandomSampler
    auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(validation_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    // Create a network
    auto net = std::make_shared<Net>(input_size, output_size,
        hidden_layer1_dimension, hidden_layer2_dimension);

    // Optimizer
    // torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::optim::Adam optimizer(net->parameters());
    float weight_decay = 0.05;
    auto options = static_cast<torch::optim::AdamOptions&>(optimizer.defaults());
    options.lr(learning_rate);
    options.weight_decay(weight_decay);

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    /* Training phase */
    std::cout << "Training...\n";

    // TODO: Random shuffle the training batches?

    Timer t_training;
    t_training.Start();

    for(int epoch=1; epoch<=num_epochs; epoch++) {
        int batch_index = 0;
        float loss_acc = 0;
        // Training loop
        net->train();
        for(auto& batch: *train_data_loader) {
            // Clear the optimizer parameters
            optimizer.zero_grad();
            auto data = batch.data;
            auto target = batch.target.squeeze();
            auto output = net->forward(data);
            auto loss = torch::nn::functional::nll_loss(output, target.detach());
            // auto loss = torch::nn::functional::nll_loss(output.log_softmax(1), target.detach());
            // Backpropagate the loss
            loss.backward();
            // Update the parameters
            optimizer.step();
            batch_index++;
            loss_acc += loss.item<float>();
        }
        std::cout << "[Training] Epoch-number: " << epoch;
        std::cout << "; training-loss: " << loss_acc / batch_index << std::endl;
        if(epoch % 10 == 0) {
            // Validation loop
            net->eval();
            float val_correct = 0;
            for (const auto& batch : *valid_data_loader) {
                auto data = batch.data;
                auto targets = batch.target.view({-1});

                auto output = net->forward(data);
                
                auto acc = (torch::exp(output)).argmax(1).eq(targets).sum();

                val_correct += acc.template item<float>();
            }
            float val_accuracy = val_correct / validation_data_size;
            if(val_accuracy >= target_val_accuracy) {
                std::cout << "Target validation accuracy of " << target_val_accuracy << " reached!\n" 
                        << "Breaking from training loop at " << epoch << " epochs...\n";
                break;
            }
            std::cout << "validation-accuracy: " << val_accuracy << std::endl;
        }
    }

    t_training.Stop();
    PrintStep("[TimingStat] Training time (s):", t_training.Seconds());

    /* Testing phase */

    // Custom data loader for testing set
    std::cout << "Loading testing dataset...\n";
    auto testing_custom_dataset = CustomDataset(
        testing_labeled_data,
        in_data_size.testing_data_size,
        node_emb,
        node_embedding_dim).map(torch::data::transforms::Stack<>());
    // int testing_batch_size = testing_custom_dataset.size().value() / num_batches;
    const size_t testing_data_size = testing_custom_dataset.size().value();
    // Sampler types: SequentialSampler, RandomSampler
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(testing_custom_dataset), 
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    std::cout << "Testing...\n";
    net->eval();

    // float /* testing_loss = 0, */ testing_acc = 0;

    Timer t_testing;
    t_testing.Start();

    size_t num_correct = 0;
    for (const auto& batch : *test_data_loader) {
        auto data = batch.data;
        auto targets = batch.target.view({-1});
        auto output = net->forward(data);
        auto prediction = output.argmax(1);
        // std::cout << "predition: " << prediction << std::endl;
        num_correct += prediction.eq(targets).sum().item<int64_t>();
    }
    t_testing.Stop();
    PrintStep("[TimingStat] Testing  time (s):", t_testing.Seconds());

    std::cout << "Testing accuracy: " << (float) num_correct/testing_data_size << std::endl;

}
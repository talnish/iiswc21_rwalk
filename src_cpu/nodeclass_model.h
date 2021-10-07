/*
 * Feed-fwd NN model for node classification.
 * Source code obtained from:
 * https://pytorch.org/cppdocs/frontend.html
 */

struct Net : torch::nn::Module {
  Net(int num_inputs, int num_outputs, int hidden_dim1, int hidden_dim2) {
    // Construct and register two Linear submodules.
    // fc1 = register_module("fc1", torch::nn::Linear(num_inputs, hidden_dim1));
    // fc2 = register_module("fc2", torch::nn::Linear(hidden_dim1, hidden_dim2));
    // fc3 = register_module("fc3", torch::nn::Linear(hidden_dim2, num_outputs));
    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(num_inputs, hidden_dim1).bias(false)));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim1, hidden_dim2).bias(false)));
    fc3 = register_module("fc3", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim2, num_outputs).bias(false)));
  }

  // Implement the Net's algorithm.
  // https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials...
  //  .../basics/feedforward_neural_network/src/neural_net.cpp
  torch::Tensor forward(torch::Tensor x) {
    // x = torch::nn::functional::relu(fc1->forward(x));
    // return torch::nn::functional::relu(fc2->forward(x));
    // x = torch::relu(torch::nn::Dropout(fc1->forward(x),0.1,false));
    x = torch::dropout(x, /*p=*/0.1, /*train=*/is_training());
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.1, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::dropout(x, /*p=*/0.1, /*train=*/is_training());
    x = torch::relu(fc3->forward(x));
    return torch::nn::functional::log_softmax(x, 1);
    // return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr} , fc3{nullptr};
};
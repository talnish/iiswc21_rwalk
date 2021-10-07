/*
 * Logistic regression model for link prediction.
 * Source code obtained from:
 * https://pytorch.org/cppdocs/frontend.html
 */

struct Net : torch::nn::Module {
  Net(int num_inputs, int num_outputs, int hidden_layer_dim) {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(num_inputs, hidden_layer_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_layer_dim, num_outputs));
  }

  // Implement the Net's algorithm.
  // https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials...
  //  .../basics/feedforward_neural_network/src/neural_net.cpp
  torch::Tensor forward(torch::Tensor x) {
    // x = torch::nn::functional::relu(fc1->forward(x));
    // return torch::nn::functional::relu(fc2->forward(x));
    x = torch::relu(fc1->forward(x));
    return torch::sigmoid(fc2->forward(x));
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

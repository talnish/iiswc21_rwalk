// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <getopt.h>

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>


/*
GAP Benchmark Suite
Class:  CLBase
Author: Scott Beamer

Handles command line argument parsing
 - Through inheritance, can add more options to object
 - For example, most kernels will use CLApp
*/


class CLBase {
 protected:
  int argc_;
  char** argv_;
  std::string name_;
  std::string get_args_ = "f:g:hk:su:";
  std::vector<std::string> help_strings_;

  int scale_ = -1;
  int degree_ = 16;
  std::string filename_ = "";
  bool symmetrize_ = false;
  bool uniform_ = false;

  void AddHelpLine(char opt, std::string opt_arg, std::string text,
                   std::string def = "") {
    const int kBufLen = 100;
    char buf[kBufLen];
    if (opt_arg != "")
      opt_arg = "<" + opt_arg + ">";
    if (def != "")
      def = "[" + def + "]";
    snprintf(buf, kBufLen, " -%c %-9s: %-54s%10s", opt, opt_arg.c_str(),
            text.c_str(), def.c_str());
    help_strings_.push_back(buf);
  }

 public:
  CLBase(int argc, char** argv, std::string name = "") :
         argc_(argc), argv_(argv), name_(name) {
    AddHelpLine('h', "", "print this help message");
    AddHelpLine('f', "file", "load graph from file");
    AddHelpLine('s', "", "symmetrize input edge list", "false");
    AddHelpLine('g', "scale", "generate 2^scale kronecker graph");
    AddHelpLine('u', "scale", "generate 2^scale uniform-random graph");
    AddHelpLine('k', "degree", "average degree for synthetic graph",
                std::to_string(degree_));
  }

  bool ParseArgs() {
    signed char c_opt;
    extern char *optarg;          // from and for getopt
    while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
      HandleArg(c_opt, optarg);
    }
    if ((filename_ == "") && (scale_ == -1)) {
      std::cout << "No graph input specified. (Use -h for help)" << std::endl;
      return false;
    }
    if (scale_ != -1)
      symmetrize_ = true;
    return true;
  }

  void virtual HandleArg(signed char opt, char* opt_arg) {
    switch (opt) {
      case 'f': filename_ = std::string(opt_arg);           break;
      case 'g': scale_ = atoi(opt_arg);                     break;
      case 'h': PrintUsage();                               break;
      case 'k': degree_ = atoi(opt_arg);                    break;
      case 's': symmetrize_ = true;                         break;
      case 'u': uniform_ = true; scale_ = atoi(opt_arg);    break;
    }
  }

  void PrintUsage() {
    std::cout << name_ << std::endl;
    // std::sort(help_strings_.begin(), help_strings_.end());
    for (std::string h : help_strings_)
      std::cout << h << std::endl;
    std::exit(0);
  }

  int scale() const { return scale_; }
  int degree() const { return degree_; }
  std::string filename() const { return filename_; }
  bool symmetrize() const { return symmetrize_; }
  bool uniform() const { return uniform_; }
};



class CLApp : public CLBase {
  bool do_analysis_ = false;
  int num_trials_ = 16;
  int64_t start_vertex_ = -1;
  bool do_verify_ = false;
  std::string graph_dir_name_ = "add-graph_dir_name";
  std::string hyperparams_file_name_ = "add-hyperparams-filename";
  int output_dim_ = 5;
  float learning_rate_ = 1;
  int num_epochs_ = 50;
  int hidden_layer_dim_ = 100;
  int hidden_layer1_dim_ = 100;
  int hidden_layer2_dim_ = 100;
  int num_batches_ = 10;
  int num_workers_ = 0;
  float training_ratio_ = 0.8;
  int node_embedding_dimension_ = 128;
  int num_walks_per_node_ = 10;
  int use_max_num_threads_ = 1;
  int num_threads_ = 48;
  int walk_length_ = 5;
  float target_val_accuracy_ = 0.75;
  int batch_size_ = 128;

 public:
  CLApp(int argc, char** argv, std::string name) : CLBase(argc, argv, name) {
    get_args_ += "c:p:l:an:r:v";
    AddHelpLine('a', "", "output analysis of last run", "false");
    AddHelpLine('n', "n", "perform n trials", std::to_string(num_trials_));
    AddHelpLine('r', "node", "start from node r", "rand");
    AddHelpLine('v', "", "verify the output of each run", "false");
    AddHelpLine('l', "", "maximum length of temporal walk", "5");
    AddHelpLine('p', "", "Labeled training file name for node classification", "add-graph-dir-name");
    AddHelpLine('c', "", "Script for configuring parameters");
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'a': do_analysis_ = true;                            break;
      case 'n': num_trials_ = atoi(opt_arg);                    break;
      case 'r': start_vertex_ = atol(opt_arg);                  break;
      case 'v': do_verify_ = true;                              break;
      case 'p': graph_dir_name_ = std::string(opt_arg);         break;
      case 'c': hyperparams_file_name_ = std::string(opt_arg);  break;
      default: CLBase::HandleArg(opt, opt_arg);
    }
  }

  bool do_analysis() const { return do_analysis_; }
  int num_trials() const { return num_trials_; }
  int64_t start_vertex() const { return start_vertex_; }
  bool do_verify() const { return do_verify_; }
  int get_output_dim() const { return output_dim_; }
  float get_learning_rate() const { return learning_rate_; }
  int get_num_epochs() const { return num_epochs_; }
  int get_hidden_layer_dim() const { return hidden_layer_dim_; }
  int get_hidden_layer1_dim() const { return hidden_layer1_dim_; }
  int get_hidden_layer2_dim() const { return hidden_layer2_dim_; }
  int get_num_batches() const { return num_batches_; }
  int get_num_workers() const { return num_workers_; }
  float get_training_ratio() const { return training_ratio_; }
  int get_node_emb_dim() const { return node_embedding_dimension_; }
  int get_num_walks_per_node() const { return num_walks_per_node_; }
  int use_max_num_threads() const { return use_max_num_threads_; }
  int get_num_threads() const { return num_threads_; }
  int get_max_walk_length() const { return walk_length_; }
  float get_target_val_accuracy() const { return target_val_accuracy_; }
  int get_batch_size() const { return batch_size_; }
  std::string get_training_file_name()  const { 
    std::string file_base_path = "../data/node_class/";
    std::string file_name = "/train.tsv";
    std::string return_file_path;
    return_file_path = file_base_path + graph_dir_name_ + file_name;
    return return_file_path; }
  std::string get_testing_file_name()  const { 
    std::string file_base_path = "../data/node_class/";
    std::string file_name = "/test.tsv";
    std::string return_file_path;
    return_file_path = file_base_path + graph_dir_name_ + file_name;
    return return_file_path; }
  std::string get_validation_file_name()  const { 
    std::string file_base_path = "../data/node_class/";
    std::string file_name = "/valid.tsv";
    std::string return_file_path;
    return_file_path = file_base_path + graph_dir_name_ + file_name;
    return return_file_path; }

  void read_params_file() // Read hyperparameter file
  {
    std::cout << "Reading hyperparameters config file " << hyperparams_file_name_ << "...\n";
    std::ifstream hyperparams_file(hyperparams_file_name_);
    if(hyperparams_file)
    {
      std::string in_line;
      while(getline(hyperparams_file, in_line))
      {
        if(in_line.find('#') == 0) continue;  // This is a comment
        else
        {
          std::string out_dim_string = "output_dim",
                      learning_rate_string = "learning_rate",
                      num_epochs_string = "num_epochs",
                      hidden_layer_dim_string = "hidden_layer_dim",
                      hidden_layer1_dim_string = "hidden_layer1_dim",
                      hidden_layer2_dim_string = "hidden_layer2_dim",
                      num_batches_string = "num_batches",
                      training_ratio_string = "training_ratio",
                      node_embedding_dim_string = "node_embedding_dim",
                      num_walks_per_node_string = "num_walks_per_node",
                      use_max_num_threads_string = "use_max_num_threads",
                      num_threads_string = "num_threads",
                      walk_length_string = "walk_length",
                      target_val_accuracy_string = "target_validation_accuracy",
                      batch_size_string = "batch_size",
                      num_workers_string = "num_workers";
          if(in_line.find(out_dim_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            output_dim_ = std::stoi(split_string[1]);
          }
          if(in_line.find(learning_rate_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            learning_rate_ = std::stof(split_string[1]);
          }
          if(in_line.find(num_epochs_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            num_epochs_ = std::stoi(split_string[1]);
          }
          if(in_line.find(hidden_layer_dim_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            hidden_layer_dim_ = std::stoi(split_string[1]);
          }
          if(in_line.find(hidden_layer1_dim_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            hidden_layer1_dim_ = std::stoi(split_string[1]);
          }
          if(in_line.find(hidden_layer2_dim_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            hidden_layer2_dim_ = std::stoi(split_string[1]);
          }
          if(in_line.find(num_batches_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            num_batches_ = std::stoi(split_string[1]);
          }
          if(in_line.find(training_ratio_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            training_ratio_ = std::stof(split_string[1]);
          }
          if(in_line.find(node_embedding_dim_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            node_embedding_dimension_ = std::stoi(split_string[1]);
          }
          if(in_line.find(num_walks_per_node_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            num_walks_per_node_ = std::stoi(split_string[1]);
          }
          if(in_line.find(use_max_num_threads_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            use_max_num_threads_ = std::stoi(split_string[1]);
          }
          if(in_line.find(num_threads_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            num_threads_ = std::stoi(split_string[1]);
          }
          if(in_line.find(walk_length_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            walk_length_ = std::stoi(split_string[1]);
          }
          if(in_line.find(target_val_accuracy_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            target_val_accuracy_ = std::stof(split_string[1]);
          }
          if(in_line.find(batch_size_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            batch_size_ = std::stoi(split_string[1]);
          }
          if(in_line.find(num_workers_string) == 0)
          {
            std::istringstream splt(in_line);
            std::vector<std::string> split_string{
              std::istream_iterator<std::string>(splt), {}
            };
            num_workers_ = std::stoi(split_string[1]);
          }

        }
      }
    }
  }
};

class CLIterApp : public CLApp {
  int num_iters_;

 public:
  CLIterApp(int argc, char** argv, std::string name, int num_iters) :
    CLApp(argc, argv, name), num_iters_(num_iters) {
    get_args_ += "i:";
    AddHelpLine('i', "i", "perform i iterations", std::to_string(num_iters_));
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'i': num_iters_ = atoi(opt_arg);            break;
      default: CLApp::HandleArg(opt, opt_arg);
    }
  }

  int num_iters() const { return num_iters_; }
};



class CLPageRank : public CLApp {
  int max_iters_;
  double tolerance_;

 public:
  CLPageRank(int argc, char** argv, std::string name, double tolerance,
             int max_iters) :
    CLApp(argc, argv, name), max_iters_(max_iters), tolerance_(tolerance) {
    get_args_ += "i:t:";
    AddHelpLine('i', "i", "perform at most i iterations",
                std::to_string(max_iters_));
    AddHelpLine('t', "t", "use tolerance t", std::to_string(tolerance_));
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'i': max_iters_ = atoi(opt_arg);            break;
      case 't': tolerance_ = std::stod(opt_arg);            break;
      default: CLApp::HandleArg(opt, opt_arg);
    }
  }

  int max_iters() const { return max_iters_; }
  double tolerance() const { return tolerance_; }
};



template<typename WeightT_>
class CLDelta : public CLApp {
  WeightT_ delta_ = 1;

 public:
  CLDelta(int argc, char** argv, std::string name) : CLApp(argc, argv, name) {
    get_args_ += "d:";
    AddHelpLine('d', "d", "delta parameter", std::to_string(delta_));
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'd':
        if (std::is_floating_point<WeightT_>::value)
          delta_ = static_cast<WeightT_>(atof(opt_arg));
        else
          delta_ = static_cast<WeightT_>(atol(opt_arg));
        break;
      default: CLApp::HandleArg(opt, opt_arg);
    }
  }

  WeightT_ delta() const { return delta_; }
};



class CLConvert : public CLBase {
  std::string out_filename_ = "";
  bool out_weighted_ = false;
  bool out_el_ = false;
  bool out_sg_ = false;

 public:
  CLConvert(int argc, char** argv, std::string name)
      : CLBase(argc, argv, name) {
    get_args_ += "e:b:w";
    AddHelpLine('b', "file", "output serialized graph to file");
    AddHelpLine('e', "file", "output edge list to file");
    AddHelpLine('w', "file", "make output weighted");
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'b': out_sg_ = true; out_filename_ = std::string(opt_arg);   break;
      case 'e': out_el_ = true; out_filename_ = std::string(opt_arg);   break;
      case 'w': out_weighted_ = true;                                   break;
      default: CLBase::HandleArg(opt, opt_arg);
    }
  }

  std::string out_filename() const { return out_filename_; }
  bool out_weighted() const { return out_weighted_; }
  bool out_el() const { return out_el_; }
  bool out_sg() const { return out_sg_; }
};

#endif  // COMMAND_LINE_H_

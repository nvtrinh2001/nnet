#pragma once
#include "matrix.hpp"
#include <math.h>
#include <random>
#include <utility>
#include <cassert>
#include <vector>

using namespace matrix;

namespace nnet {

  template<typename T>
  class MLP {

    public:
      // a list contains #nodes of each layer
      std::vector<size_t> units_per_layer;
      std::vector<Matrix<T>> bias_vectors;
      std::vector<Matrix<T>> weight_matrices;
      std::vector<Matrix<T>> activations;

      float learning_rate;

      // explicit: not allow: MLP<float> a = ...; instead: a(...)
      explicit MLP(std::vector<size_t> units_per_layer, float lr = 0.001f) :
        units_per_layer(units_per_layer),
        bias_vectors(),
        weight_matrices(),
        activations(),
        learning_rate(lr) {
          
          for (int layer_idx = 0; layer_idx < units_per_layer.size() - 1; layer_idx++) {
            // output of 1 layer is the input of the next layer
            size_t in_channels = units_per_layer[layer_idx];
            size_t out_channels = units_per_layer[layer_idx + 1];

            // init random Gaussian for weights and biases
          
            // eg: 
            // input:  3 nodes
            // hidden: 4 nodes --> dimension: 4x3
            auto W = mtx<T>::random(out_channels, in_channels);
            weight_matrices.push_back(W);

            auto b = mtx<T>::random(out_channels, 1);
            bias_vectors.push_back(b);

            activations.resize(units_per_layer.size());
          }

        }

      // Each layer of the neural network will be of the form: 
      // output <- sigmoid( Weight.matmul( input ) + bias )
      
      // activation function
      static auto sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
      }

      auto forward(Matrix<T> input) {
        assert(get<0>(input.shape) == units_per_layer[0] && get<1>(input.shape));

        // add input to the first value of activations
        activations[0] = input;
        Matrix prev(input);

        for (int layer_idx = 0; layer_idx < units_per_layer.size() - 1; layer_idx++) {
          Matrix out = weight_matrices[layer_idx].matmul(prev) + bias_vectors[layer_idx];
          out = out.apply_function(sigmoid);

          activations[layer_idx + 1] = out;
          prev = out;
        }

        return prev;
      }

      static auto d_sigmoid(float x) {
        return x * (1 - x);
      }

      void backprop(Matrix<T> target) {
        assert(get<0>(target.shape) == units_per_layer.back());

        // determine the simple error
        // error = target - output
        auto y = target;
        auto y_hat = activations.back();
        auto error = y - y_hat;

        // backprob output -> input and step the weights
        for (int i = weight_matrices.size() - 1; i >= 0; i--) {
          // calculate errors for previous layer
          auto Wt = weight_matrices[i].transpose();
          auto prev_errors = Wt.matmul(error);

          // apply derivative of function evaluated at activations
          auto d_outputs = activations[i + 1].apply_function(d_sigmoid);
          auto gradients = error.matmul_elementwise(d_outputs);
          gradients = gradients.matmul_scalar(learning_rate);

          // backprop for weights
          auto activations_transposed = activations[i].transpose();
          auto weight_gradients = gradients.matmul(activations_transposed);

          // adjust weights
          bias_vectors[i] = bias_vectors[i] + gradients;
          weight_matrices[i] = weight_matrices[i] + weight_gradients;
          error = prev_errors;
        }
      }

  };
}

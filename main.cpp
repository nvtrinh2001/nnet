#include "mlp.hpp"
#include <fstream>
#include <deque>

// helper to initialize multi-layer perceptron with n hidden layers each w/ same num hidden units
auto make_model(size_t in_channels,
                          size_t out_channels,
                          size_t hidden_units_per_layer,
                          int hidden_layers,
                          float lr) {
  std::vector<size_t> units_per_layer;

  units_per_layer.push_back(in_channels);

  for (int i = 0; i < hidden_layers; ++i)
    units_per_layer.push_back(hidden_units_per_layer);

  units_per_layer.push_back(out_channels);

  nnet::MLP<float> model(units_per_layer, 0.01f);
  return model;
}

template<typename T>
void log(std::ofstream &file, const Matrix<T> &x, const Matrix<T> &y, const Matrix<T> &y_hat) {
  auto mse = (y.data[0] - y_hat.data[0]);
  mse = mse * mse;

  file << mse << " "
       << x.data[0] << " "
       << y.data[0] << " "
       << y_hat.data[0] << " \n";
}

int main() {

  std::srand(42069);

  // init model
  int in_channels{1}, out_channels{1}, hidden_units_per_layer{8}, hidden_layers{3};
  float lr{.5f};

  auto model = make_model(
      in_channels=1,
      out_channels=1,
      hidden_units_per_layer=8,
      hidden_layers=3,
      lr=.5f);

  // train
  std::ofstream my_file;
  my_file.open ("data.txt");
  int max_iter{1000};
  float mse;
  auto deque = std::deque<float>(max_iter);
  for(int i = 1; i <= max_iter; ++i) {

    // generate (x, y) training data
    auto x = mtx<float>::random(in_channels, 1).matmul_scalar(3.);
    auto y = x.apply_function([](float v) -> float { 
        return sin(v) * sin(v); 
    });

    auto y_hat = model.forward(x);  // forward pass
    model.backprop(y); // backward pass

    // compute and print error
    mse = (y - y_hat).square().data[0];
    deque.push_back(mse);
    log(my_file, x, y, y_hat);

  }

  my_file.close();
}


#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace matrix {

template<typename T>
class Matrix {

  size_t cols;
  size_t rows;

  public:
  std::vector<T> data;
  std::tuple<size_t, size_t> shape; // shape of the matrix
  int num_elements;

  // constructors
  Matrix(size_t new_rows, size_t new_cols) : 
    rows(new_rows), 
    cols(new_cols), 
    shape(new_rows, new_cols), 
    num_elements(new_rows * new_cols), 
    data({}) {
    data.resize(new_rows * new_cols, T());
  }

  Matrix() : rows(0), cols(0), data({}), shape(0, 0) {}

  // print shape
  void get_shape() {
    std::cout << "Matrix size: (" << rows << "," << cols << ")\n";
  }

  // print matrix
  void print() {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        std::cout << (*this)(i, j) << " ";
      }

      std::cout << "\n";
    }
  }

  // access element
  T& operator()(size_t row, size_t col) {
    return data[row * cols + col];
  }

  // matrix multiplication
  // Time:  O(n^3)
  // Space: O(n^2)
  Matrix matmul(Matrix &target) {
    // eg: 
    // 1 1 1     1 3 5 1 4
    // 2 2 2  *  3 4 4 1 6 --> (2x3 * 3x5 = 2x5)
    //           4 1 2 5 2
    assert(cols == target.rows);
    Matrix output(rows, target.cols);

    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < target.cols; c++) {
        
        // can also be cols as they are the same
        for (size_t k = 0; k < target.rows; k++) { 
          output(r, c) += (*this)(r, k) * target(k, c);
        }

      }
    }

    return output;
  }

  // matrix multiplication with the same shape
  Matrix matmul_elementwise(Matrix &target) {
    assert(shape == target.shape);
    Matrix output((*this)); // copy current matrix

    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < cols; c++) {
        output(r, c) = (*this)(r, c) * target(r, c);
      }
    }

    return output;
  }

  // square of a matrix
  Matrix square() {
    Matrix output((*this));

    output = matmul_elementwise(output);
    
    return output;
  }

  // scalar mulitiplication
  Matrix matmul_scalar(T scalar) {
    Matrix output((*this));
    for (size_t r = 0; r < output.rows; r++) {
      for (size_t c = 0; c < output.cols; c++) {
        output(r, c) = scalar * (*this)(r, c);
      }
    }

    return output;
  }

  // matrix addition
  Matrix add(Matrix &target) {
    assert(shape == target.shape);
    // Matrix output((*this)); or
    Matrix output(rows, get<1>(target.shape));

    for (size_t r = 0; r < output.rows; r++) {
      for (size_t c = 0; c < output.cols; c++) {
        output(r, c) = (*this)(r, c) + target(r, c);
      }
    }

    return output;
  }

  Matrix operator+(Matrix &target) {
    return add(target);
  }

  // matrix negation
  Matrix operator-() {
    Matrix output(rows, cols);
    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < cols; c++) {
        output(r, c) = -(*this)(r, c);
      }
    }

    return output;
  }

  // matrix substraction
  Matrix operator-(Matrix &target) {
    assert(shape == target.shape);
    Matrix output = -target;

    output = add(output);

    return output;
  }

  // matrix transpose
  Matrix transpose() {
    size_t new_rows = cols;
    size_t new_cols = rows;
    Matrix transposed(new_rows, new_cols);
  
    for (size_t r = 0; r < transposed.rows; r++) {
      for (size_t c = 0; c < transposed.cols; c++) {
        transposed(r, c) = (*this)(c, r);
      }
    }

    return transposed;
  }

  // introduce non-linearities
  Matrix apply_function(const std::function<T(const T &)> &func) {
    Matrix output((*this));

    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < cols; c++) {
        output(r, c) = func((*this)(r, c)); // function accepts each element as a param
      }
    }

    return output;
  }
 
};

template<typename T>
struct mtx {
  static Matrix<T> random(size_t rows, size_t cols) {
    Matrix<T> M(rows, cols);

    std::random_device rd{};
    std::mt19937 gen{rd()};

    // init Gaussian distribution with N(mean=0, std=1/sqrt(num_elements))
    T num_elements(M.num_elements);
    T std(1 / sqrt(num_elements));
    std::normal_distribution<T> dist(0, std);

    // fill in the elements
    for (size_t r = 0; r < rows; r++) {
      for (size_t c = 0; c < cols; c++) {
        M(r, c) = dist(gen);
      }
    }

    return M;
  }
};

};

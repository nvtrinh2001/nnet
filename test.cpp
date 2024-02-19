#include "matrix.hpp"

int test(int /*argc*/, char * /*argv*/ []) {

  auto M = matrix::mtx<float>::random(2, 2); // init randn matrix

  M.get_shape();

  std::cout << "\nMatrix:\n";
  M.print(); // print the OG matrix

  std::cout << "\nSubtraction:\n";
  (M-M).print();  // print M minus itself

  std::cout << "\nAddition:\n";
  (M+M).print();  // print its sum

  std::cout << "\nScalar with 2:\n";
  (M.matmul_scalar(2.f)).print();  // print 2x itself

  std::cout << "\nMultiply with itself:\n";
  (M.matmul_elementwise(M)).print(); // mult M w itself

  std::cout << "\nTranspose:\n";
  auto MT = M.transpose(); // transpose the matrix
  MT.print();

  std::cout << "\nMultiply 2 matrices:\n";
  (MT.matmul(M)).print();  // form symm. pos. def. matrix

  std::cout << "\nUsing function that subtracts itself:\n";
  (M.apply_function([](auto x){return x-x;} )).print(); // apply fun

  return 0;
}


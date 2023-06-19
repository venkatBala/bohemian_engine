#include <iostream>

#include "bhime.hpp"

auto const N = 5;
constexpr auto NX = 2 * 1920;
constexpr auto NY = 2 * 1080;

int main(int argc, char **argv) {
  // 2D density map
  Histogram2d<NX, NY> eigenvalue_density(-4, 4, -4, 4);
  BHMatrixGenerator<N, 4000000, -1, 0, 1> generator;
  std::for_each(generator.begin(), generator.end(), [&](const auto &matrix) {
    if (!matrix.is_singular()) {
      auto eigenvalues = matrix.eigenvalues();
      for (auto i = 0; i < N; i++) {
        eigenvalue_density.update(eigenvalues[i].real(), eigenvalues[i].imag());
      }
    }
  });

  // Save density map as image
  eigenvalue_density.to_png("test.png");

  return 0;
}

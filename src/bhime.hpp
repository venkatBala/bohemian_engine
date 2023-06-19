#include <gsl/gsl_histogram2d.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <png++/png.hpp>
#include <png++/rgb_pixel.hpp>
#include <random>

// Int vector typedef
using ivec = std::vector<int>;

template <int N>
class BHMatrix {
 public:
  typedef double ElementType;
  using value_type = ElementType;
  using reference_type = ElementType &;
  using pointer_type = ElementType *;

  int row_id = 0;
  int col_id = 0;

  // Default ctor
  BHMatrix() : size_(N), data_(Eigen::Matrix<ElementType, N, N>()) {
    data_.setZero();
  }

  // Copy constructor
  BHMatrix(const BHMatrix &matrix) : size_(matrix.size_), data_(matrix.data_) {}

  [[nodiscard]] inline value_type operator()(int row, int col) const noexcept {
    return data_(row, col);
  }

  [[nodiscard]] inline reference_type operator()(int row, int col) noexcept {
    return data_(row, col);
  }

  friend std::ostream &operator<<(std::ostream &out, const BHMatrix &matrix) {
    out << matrix.data_;
    return out;
  }

  bool is_singular() const noexcept {
    return (data_.determinant() == 0) ? true : false;
  }

  auto eigenvalues() const noexcept { return data_.eigenvalues(); }

 private:
  const int size_;
  Eigen::Matrix<double, N, N> data_;
};

template <int MatrixSize, int NumMatrices, int... values>
class BHMatrixGenerator {
 public:
  using iterator_tag = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = int;
  using reference_type = int &;
  using pointer = int *;

  BHMatrixGenerator()
      : counter_(0),
        iset_({values...}),
        seed_(std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()),
        generator_(std::mt19937(seed_)),
        distr_(0, iset_.size() - 1) {}

  explicit BHMatrixGenerator(int count)
      : counter_(count),
        iset_({values...}),
        seed_(std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count()),
        generator_(std::mt19937(seed_)),
        distr_(0, iset_.size() - 1) {}

  explicit BHMatrixGenerator(const BHMatrixGenerator &sentinel)
      : counter_(sentinel.counter_),
        iset_(sentinel.iset_),
        seed_(sentinel.seed_),
        max_limit_(sentinel.max_limit_),
        generator_(sentinel.generator_),
        distr_(sentinel.distr_),
        current_matrix_(sentinel.current_matrix_) {}

  [[nodiscard]] inline auto &operator*() const noexcept {
    return current_matrix_;
  }

  [[nodiscard]] inline auto operator->() noexcept { return &current_matrix_; }

  auto &operator++() noexcept {
    for (auto row = 0; row < MatrixSize; row++) {
      for (auto col = 0; col < MatrixSize; col++) {
        auto random_index = distr_(generator_);
        current_matrix_(row, col) = iset_[random_index];
      }
    }

    counter_++;

    return *this;
  }

  BHMatrixGenerator begin() const noexcept { return BHMatrixGenerator(); }

  BHMatrixGenerator end() const noexcept {
    return BHMatrixGenerator(NumMatrices + 1);
  }

  friend bool operator==(const BHMatrixGenerator &lhs,
                         const BHMatrixGenerator &rhs) {
    return lhs.counter_ == rhs.counter_;
  }

  friend bool operator!=(const BHMatrixGenerator &lhs,
                         const BHMatrixGenerator &rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const BHMatrixGenerator &lhs,
                        const BHMatrixGenerator &rhs) {
    return (lhs.counter_ < rhs.counter_);
  }

  friend bool operator>(const BHMatrixGenerator &lhs,
                        const BHMatrixGenerator &rhs) {
    return (rhs.counter_ < lhs.counter_);
  }

  friend bool operator>=(const BHMatrixGenerator &lhs,
                         const BHMatrixGenerator &rhs) {
    return !(lhs < rhs);
  }

  friend bool operator<=(const BHMatrixGenerator &lhs,
                         const BHMatrixGenerator &rhs) {
    return !(rhs < lhs);
  }

 private:
  value_type counter_;
  BHMatrix<MatrixSize> current_matrix_;
  const ivec iset_;
  const int max_limit_ = NumMatrices;
  unsigned int seed_;
  std::mt19937 generator_;
  std::uniform_int_distribution<int> distr_;
};

template <int MatrixSize, int NumMatrices, int... values>
class TridiagonalBHMatrixGenerator
    : public BHMatrixGenerator<MatrixSize, NumMatrices, values...> {
 public:
  TridiagonalBHMatrixGenerator()
      : counter_(0),
        iset_({values...}),
        generator_(std::mt19937()),
        distr_(0, iset_.size() - 1) {}

  explicit TridiagonalBHMatrixGenerator(int counter)
      : counter_(counter),
        iset_({values...}),
        generator_(std::mt19937()),
        distr_(0, iset_.size() - 1) {}

  explicit TridiagonalBHMatrixGenerator(
      const TridiagonalBHMatrixGenerator &matrix)
      : counter_(matrix.counter_),
        current_matrix_(matrix.current_matrix_),
        iset_(matrix.iset_) {}

  auto &operator++() {
    // sub diagonal entires
    auto sub_diagonal = current_matrix_.diagonal(-1);
    std::for_each(sub_diagonal.begin(), sub_diagonal.end(),
                  [&](auto &element) { element = iset_[distr_(generator_)]; });

    // Principal diagonal
    auto principal_diag = current_matrix_.diagonal(0);
    std::for_each(principal_diag.begin(), principal_diag.end(),
                  [&](auto &element) { element = iset_[distr_(generator_)]; });

    // super diagonal
    auto super_diagonal = current_matrix_.diagonal(1);
    std::for_each(super_diagonal.begin(), super_diagonal.end(),
                  [&](auto &element) { element = iset_[distr_(generator_)]; });

    counter_++;
    return *this;
  }

 private:
  typename BHMatrixGenerator<MatrixSize, NumMatrices, values...>::value_type
      counter_;
  BHMatrix<MatrixSize> current_matrix_;
  const ivec iset_;
  std::mt19937 generator_;
  std::uniform_int_distribution<int> distr_;
};

// 2-D historgram
template <int NX, int NY>
class Histogram2d {
 public:
  Histogram2d(double xlo, double xhi, double ylo, double yhi)
      : xbins_(NX), ybins_(NY), data_(gsl_histogram2d_alloc(NX, NY)) {
    gsl_histogram2d_set_ranges_uniform(data_, xlo, xhi, ylo, yhi);
  }

  void update(double x, double y) { gsl_histogram2d_increment(data_, x, y); }

  ~Histogram2d() { gsl_histogram2d_free(data_); }

  double operator()(int i, int j) const noexcept {
    return gsl_histogram2d_get(data_, i, j);
  }

  void scale() {
    auto max = gsl_histogram2d_max_val(data_);
    gsl_histogram2d_scale(data_, (double)(1.0 / max));
  }

  void to_png(const char *filename) {
    // Save the density map to png
    png::image<png::rgb_pixel> image(xbins_, ybins_);

    for (auto y = 0; y < image.get_height(); y++) {
      for (auto x = 0; x < image.get_width(); x++) {
        auto density = gsl_histogram2d_get(data_, x, y);

        image[y][x] = (density == 0) ? png::rgb_pixel(0, 0, 0)
                                     : png::rgb_pixel(255, 0, 0);
      }
    }

    image.write(filename);
  }

 private:
  gsl_histogram2d *data_;
  const int xbins_;
  const int ybins_;
};

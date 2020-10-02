#include "blas_wrapper_fixture.h"
#include "gtest/gtest.h"
#include "test_helper.h"

namespace {
template <class T>
class dgekvTest : public testing::Test {
 public:
  void check(const std::vector<double>& v, double a, double b) {
    auto& f = this->blas_wrapper_fixture;
    for (int row = 0; row < f.n; row++) {
      for (int idx = 0; idx < f.inc_z; idx++) {
        auto value = a;
        if (idx == f.offset_z) {
          value = b;
        }
        ASSERT_EQ(v[row * f.inc_z + idx], value) << "v[" << row << " * f.inc_z + " << idx << "]\n";
      }
    }
  }
  T blas_wrapper_fixture;
};

using Implementations = testing::Types<
    Blas_Wrapper_Fixture<std::vector, std::allocator>,
#ifdef HAVE_CUDA
    Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator>
#endif
    >;
TYPED_TEST_SUITE(dgekvTest, Implementations);

TYPED_TEST(dgekvTest, Test) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n, 1.0, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n, 2.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n, 1.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      2.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      -4.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), 1.0, 0.0);
}

TYPED_TEST(dgekvTest, 0bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n,  2.0, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n, -2.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n,  1.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      2.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      0.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), 1.0, -8.0);
}

TYPED_TEST(dgekvTest, 1bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n,  1.0, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n,  2.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n, -5.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      2.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      1.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), -5.0, -1.0);
}

TYPED_TEST(dgekvTest, 1aTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n, 0.5, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n, 8.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n, 1.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      1.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      -4.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), 1.0, 0.0);
}

TYPED_TEST(dgekvTest, 1a0bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n,  2.0, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n, -2.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n,  1.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      1.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      0.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), 1.0, -4.0);
}

TYPED_TEST(dgekvTest, 1a1bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.vfill(  f.n,  1.0, f.X, f.offset_x, f.inc_x);
  f.blas_wrapper.vfill(  f.n,  3.0, f.Y, f.offset_y, f.inc_y);
  f.blas_wrapper.vfill(f.inc_z * f.n, -4.0, f.Z, 1);
  f.blas_wrapper.dgekv(
      f.n,
      1.0,
      f.X, f.offset_x, f.inc_x,
      f.Y, f.offset_y, f.inc_y,
      1.0,
      f.Z, f.offset_z, f.inc_z);
  this->check(get_vector(f.Z), -4.0, -1.0);
}
}  // namespace

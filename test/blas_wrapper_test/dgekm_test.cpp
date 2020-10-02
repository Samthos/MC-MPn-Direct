#include "blas_wrapper_fixture.h"
#include "gtest/gtest.h"
#include "test_helper.h"

namespace {
template <class T>
class dgekmTest : public testing::Test {
 public:
  void check(const std::vector<double>& v, double a, double b) {
    auto& f = this->blas_wrapper_fixture;
    for (int col = 0; col < f.n; col++) {
      int row = 0;
      for (; row < f.offset_c; row++) {
        ASSERT_EQ(v[col * f.ldc + row], a);
      }
      for (; row < f.offset_c + f.m; row++) {
        ASSERT_EQ(v[col * f.ldc + row], b);
      }
      for (; row < f.ldc; row++) {
        ASSERT_EQ(v[col * f.ldc + row], a);
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
TYPED_TEST_SUITE(dgekmTest, Implementations);

TYPED_TEST(dgekmTest, Test) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n, 1.0, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n, 2.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n, 1.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      2.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      -4.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), 1.0, 0.0);
}

TYPED_TEST(dgekmTest, 0bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n,  2.0, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n, -2.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n,  1.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      2.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      0.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), 1.0, -8.0);
}

TYPED_TEST(dgekmTest, 1bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n,  1.0, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n,  2.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n, -5.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      2.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      1.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), -5.0, -1.0);
}

TYPED_TEST(dgekmTest, 1aTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n, 0.5, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n, 8.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n, 1.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      1.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      -4.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), 1.0, 0.0);
}

TYPED_TEST(dgekmTest, 1a0bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n,  2.0, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n, -2.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n,  1.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      1.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      0.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), 1.0, -4.0);
}

TYPED_TEST(dgekmTest, 1a1bTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(  f.m, f.n,  1.0, f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(  f.m, f.n,  3.0, f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(f.ldc, f.n, -4.0, f.C, f.ldc);
  f.blas_wrapper.dgekm(
      false, false,
      f.m, f.n,
      1.0,
      f.A, f.offset_a, f.lda,
      f.B, f.offset_b, f.ldb,
      1.0,
      f.C, f.offset_c, f.ldc);
  this->check(get_vector(f.C), -4.0, -1.0);
}
}  // namespace

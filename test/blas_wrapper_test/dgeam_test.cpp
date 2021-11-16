#include "blas_wrapper_fixture.h"
#include "test_helper.h"
#include "gtest/gtest.h"

namespace {
template <class T>
class dgeamTest : public testing::Test {
 public:
  T blas_wrapper_fixture;
};

using Implementations = testing::Types<
    Blas_Wrapper_Fixture<std::vector, std::allocator>,
    Blas_Wrapper_Fixture<std::vector, std::allocator>>;
TYPED_TEST_SUITE(dgeamTest, Implementations);

TYPED_TEST(dgeamTest, dgeamTest) {
  auto& f = this->blas_wrapper_fixture;
  f.blas_wrapper.mfill(
      f.m, f.n,
      1.0,
      f.A, f.offset_a, f.lda);
  f.blas_wrapper.mfill(
      f.m, f.n,
      2.0,
      f.B, f.offset_b, f.ldb);
  f.blas_wrapper.mfill(
      f.ldc, f.n,
      1.0,
      f.C, f.ldc);
  f.blas_wrapper.dgeam(
      false, false,
      f.m, f.n,
      2.0,
      f.A, f.offset_a, f.lda,
      -1.0,
      f.B, f.offset_b, f.ldb,
      f.C, f.offset_c, f.ldc);

  std::vector<double> h_C = get_vector(f.C);
  for (int col = 0; col < f.n; col++) {
    int row = 0;
    for (; row < f.offset_c; row++) {
      ASSERT_EQ(h_C[col * f.ldc + row], 1.0);
    }
    for (; row < f.offset_c + f.m; row++) {
      ASSERT_EQ(h_C[col * f.ldc + row], 0.0);
    }
    for (; row < f.ldc; row++) {
      ASSERT_EQ(h_C[col * f.ldc + row], 1.0);
    }
  }
}
}  // namespace

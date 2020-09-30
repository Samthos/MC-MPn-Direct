#include "gtest/gtest.h"
#include "blas_wrapper.h"
#include "../test_helper.h"

namespace {
  template <template <class, class> class Container, template <class> class Allocator>
  class Blas_Wrapper_Fixture {
    public:
      typedef Container<double, Allocator<double>> vector_double;
      typedef Blas_Wrapper<Container, Allocator> Blas_Wrapper_Type;

      Blas_Wrapper_Fixture() : 
        m(10), n(10),
        lda(20), ldb(20), ldc(20),
        offset_a(1), offset_b(2), offset_c(3),
        A(lda * n, 0.0),
        B(ldb * n, 0.0),
        C(ldc * n, 0.0) { }

      size_t m;
      size_t n;
      size_t lda;
      size_t ldb;
      size_t ldc;
      size_t offset_a;
      size_t offset_b;
      size_t offset_c;
      vector_double A;
      vector_double B;
      vector_double C;
      Blas_Wrapper_Type blas_wrapper;
  };

  template class Blas_Wrapper_Fixture<std::vector, std::allocator>;
  typedef Blas_Wrapper_Fixture<std::vector, std::allocator> Blas_Wrapper_Host_Fixture;

  template class Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator>;
  typedef Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator> Blas_Wrapper_Device_Fixture;

  template <class T>
  class BlasWrapperTest : public testing::Test {
   public:
    T blas_wrapper_fixture;
  };

  using Implementations = testing::Types<Blas_Wrapper_Host_Fixture>; // , Blas_Wrapper_Device_Fixture>;
  TYPED_TEST_SUITE(BlasWrapperTest, Implementations);

  void pmat(size_t m, size_t n, 
      const std::vector<double>& A, size_t lda) {
    for (int row = 0; row < m; row++) {
      for (int col = 0; col < n; col++) {
        printf("%12.6f", A[col * lda + row]);
      }
      printf("\n");
    }
    printf("\n");
  }

  TYPED_TEST(BlasWrapperTest, dgeamTest) {
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
        f.C, f.offset_c, f.ldc);
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
}

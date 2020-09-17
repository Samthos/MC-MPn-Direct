#include <thrust/device_vector.h>
#include <vector>

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
        n(20),
        A(n * n),
        B(n * n),
        C(n * n),
        x(n),
        y(n)
    {
#ifdef HAVE_CUDA
      namespace NS = thrust;
#else
      namespace NS = std;
#endif
      auto v = make_psi(n, n, 1.0);
      NS::copy(v.begin(), v.end(), A.begin());
      NS::copy(v.begin(), v.end(), B.begin());
      NS::copy(v.begin(), v.begin() + n, x.begin());
    }

      size_t n;
      vector_double A;
      vector_double B;
      vector_double C;
      vector_double x;
      vector_double y;
      Blas_Wrapper_Type blas_wrapper;
  };

  template class Blas_Wrapper_Fixture<std::vector, std::allocator>;
  typedef Blas_Wrapper_Fixture<std::vector, std::allocator> Blas_Wrapper_Host_Fixture;

/*
  template class Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator>;
  typedef Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator> Blas_Wrapper_Device_Fixture;
*/

  template <class T>
  class BlasWrapperTest : public testing::Test {
   public:
    void check(int sign, int start, int stop, int n_tau, const std::vector<double>& array, const std::string& array_name) {
      // double polygamma_factor = sign * PolyGamma_Difference(start, stop, n_tau+1);
      // for (int row = 0; row < electron_pairs; row++) {
      //   for (int col = 0; col < electron_pairs; col++) {
      //     ASSERT_FLOAT_EQ(array[col * electron_pairs + row], polygamma_factor* value(row, col))
      //       << "row = " << row << ", col = " << col << " of " << array_name << "\n";
      //   }
      // }
    }

    T blas_wrapper_fixture;
  };

  using Implementations = testing::Types<Blas_Wrapper_Host_Fixture>; // , Blas_Wrapper_Device_Fixture>;
  TYPED_TEST_SUITE(BlasWrapperTest, Implementations);

  TYPED_TEST(BlasWrapperTest, ddotTest) {
    this->blas_wrapper_fixture.blas_wrapper.ddot();
  }
}

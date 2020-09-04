#include <thrust/device_vector.h>
#include <memory>
#include "gtest/gtest.h"

#include "../test_helper.h"

#include "../../src/basis/dummy_movec_parser.h"
#include "../../src/basis/wavefunction.h"

namespace {
  std::vector<double> create_ao_amplitudes(int electrons, int n_basis_functions) {
    std::vector<double> ao_amplitudes(electrons * n_basis_functions);
    for (int electron = 0; electron < electrons; electron++) {
      for (int basis_function = 0; basis_function < n_basis_functions; basis_function++) {
        ao_amplitudes[electron * n_basis_functions + basis_function] = 1.0 / ((electron+1) * (electron+1) * (basis_function+1) * (basis_function+1));
      }
    }
    return ao_amplitudes;
  }

  template <template <class, class> class Container, template <class> class Allocator>
  class WavefunctionFixture {
    typedef Wavefunction<Container, Allocator> Wavefunction_Type;
    typedef Container<double, Allocator<double>> vector_double;
    typedef Container<Point, Allocator<Point>> vector_Point;

    public:
      WavefunctionFixture() :
        pos(10),
        movecs(new Dummy_Movec_Parser()),
        wavefunction(&pos, movecs),
        ao_amplitudes(create_ao_amplitudes(wavefunction.electrons, wavefunction.n_basis_functions)) 
      {
      }

      vector_Point pos;
      std::shared_ptr<Movec_Parser> movecs;
      Wavefunction_Type wavefunction;
      vector_double ao_amplitudes;
  };

  template <class T>
  class WavefunctionTest : public testing::Test {
   public:
    T fixture;
  };

  using Implementations = testing::Types<
    WavefunctionFixture<std::vector, std::allocator>,
    WavefunctionFixture<thrust::device_vector, thrust::device_allocator>
        >;
  TYPED_TEST_SUITE(WavefunctionTest, Implementations);

  TYPED_TEST(WavefunctionTest, AOtoMO) {
    this->fixture.wavefunction.ao_to_mo(this->fixture.ao_amplitudes);

    auto psi = get_vector(this->fixture.wavefunction.psi);

    double polygamma_factor = PolyGamma_Difference(0, this->fixture.wavefunction.n_basis_functions, 2);
    for (int orbital = 0; orbital < this->fixture.wavefunction.lda; orbital++) {
      for (int electron = 0; electron < this->fixture.wavefunction.electrons; electron++) {
        ASSERT_FLOAT_EQ(
            psi[electron * this->fixture.wavefunction.lda + orbital], 
            polygamma_factor / ((orbital+1.0) * (electron + 1.0) * (electron + 1.0))
        ) << orbital << ", " << electron;
      }
    }
  }
}

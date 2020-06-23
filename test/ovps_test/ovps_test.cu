//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include "../../src/qc_ovps.h"
#include "../../src/basis/wavefunction.h"

namespace {
  template <class T>
  class ovpsTest : public testing::Test {
   public:
    void SetUp() override {
      int electron_pairs = 10;
      init_wavefunction(psi1, electron_pairs,  1);
      init_wavefunction(psi2, electron_pairs, -1);
      ovps.init(2, electron_pairs);
    }

    void init_wavefunction(Wavefunction& psi, int electron_pairs, int sign) {
      psi.iocc1 = 1;
      psi.iocc2 = 6;
      psi.ivir1 = 6;
      psi.ivir2 = 16;
      psi.number_of_molecuar_orbitals = 16;
      psi.electrons = electron_pairs;
      psi.lda = psi.ivir2;

      psi.psi.resize(psi.lda * psi.electrons);
      psi.psiTau.resize(psi.lda * psi.electrons);

      for (int col = 0, idx = 0; col < psi.electrons; col++) {
        for (int row = 0; row < psi.lda; row++, idx++) {
          psi1.psi[idx] =  sign * idx;
        }
      }
    }

    Wavefunction psi1;
    Wavefunction psi2;
    OVPS<T> ovps;
  };

  using Implementations = testing::Types<std::vector<double>>; // , thrust::device_vector<double>>;
  TYPED_TEST_SUITE(ovpsTest, Implementations);

//  TYPED_TEST(ovpsTest, update) {
//  }
}

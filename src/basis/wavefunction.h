#ifndef WAVEFUNCTION_H_
#define WAVEFUNCTION_H_

#include <vector>
#include <array>
#include "nw_vectors.h"

namespace WS {
  enum Wavefunction_Sources {
    electron_pairs_1 = 0b0000,
    electron_pairs_2 = 0b0001,
    electrons        = 0b0010,
    mask             = 0b0011,
  };
}

namespace WT {
  enum Wavefunction_Types {
    normal = 0b0000,
    dx     = 0b0100,
    dy     = 0b1000,
    dz     = 0b1100,
    mask   = 0b1100,
  };
}

namespace WC {
  enum Wavefunction_Code {
    electron_pairs_1 = WS::electron_pairs_1,
    electron_pairs_2 = WS::electron_pairs_2,
    electrons        = WS::electrons,
    electron_pairs_1_dx = WT::dx | WS::electron_pairs_1,
    electron_pairs_2_dx = WT::dx | WS::electron_pairs_2,
    electrons_dx        = WT::dx | WS::electrons,
    electron_pairs_1_dy = WT::dy | WS::electron_pairs_1,
    electron_pairs_2_dy = WT::dy | WS::electron_pairs_2,
    electrons_dy        = WT::dy | WS::electrons,
    electron_pairs_1_dz = WT::dz | WS::electron_pairs_1,
    electron_pairs_2_dz = WT::dz | WS::electron_pairs_2,
    electrons_dz        = WT::dz | WS::electrons,
  };
}

namespace WM {
  enum Wavefunction_Molecules {
    primary   = 0b000000,
    monomer_a = 0b010000,
    monomer_b = 0b100000,
    mask      = 0b110000,
  };
}

class Wavefunction {
 public:
  Wavefunction() {}
  Wavefunction(std::vector<std::array<double, 3>>* p, NWChem_Movec_Parser&);

  const double *data() const;
  const double *occ() const;
  const double *vir() const;
  double *dataTau();
  double *occTau();
  double *virTau();

  size_t iocc1;
  size_t iocc2;
  size_t ivir1;
  size_t ivir2;
  size_t number_of_molecuar_orbitals;

  size_t electrons;

  size_t lda;
  size_t rows;
  size_t col;
  // type for row/col major

  std::vector<double> psi;
  std::vector<double> psiTau;
  std::vector<double> movecs;
  std::vector<std::array<double, 3>>* pos;

 private:
};
#endif  // WAVEFUNCTION_H_

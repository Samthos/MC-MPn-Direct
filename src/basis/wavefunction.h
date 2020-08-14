#ifndef WAVEFUNCTION_H_
#define WAVEFUNCTION_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif 
#include <vector>
#include <array>
#include "nwchem_movec_parser.h"

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

template <template <class, class> class Container, template <class> class Allocator>
class Wavefunction {
 public:
  Wavefunction() {}
  Wavefunction(std::vector<Point>* p, const std::shared_ptr<Movec_Parser>);

  double *data();
  double *occ();
  double *vir();
  double *dataTau();
  double *occTau();
  double *virTau();

  const double *data() const;
  const double *occ() const;
  const double *vir() const;
  const double *dataTau() const;
  const double *occTau() const;
  const double *virTau() const;

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

  Container<double, Allocator<double>> psi;
  Container<double, Allocator<double>> psiTau;
  Container<double, Allocator<double>> movecs;
  std::vector<Point>* pos;

 private:
  static double* get_raw_pointer(Container<double, Allocator<double>>&);
  static const double* get_raw_pointer(const Container<double, Allocator<double>>&);
};

template <> double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(std::vector<double, std::allocator<double>>&);
template <> const double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(const std::vector<double, std::allocator<double>>&);
template class Wavefunction<std::vector, std::allocator>;
typedef Wavefunction<std::vector, std::allocator> Wavefunction_Host;

#ifdef HAVE_CUDA
template <> double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(thrust::device_vector<double, thrust::device_allocator<double>>&);
template <> const double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(const thrust::device_vector<double, thrust::device_allocator<double>>&);
template class Wavefunction<thrust::device_vector, thrust::device_allocator>;
typedef Wavefunction<thrust::device_vector, thrust::device_allocator> Wavefunction_Device;
#endif  // HAVE_CUDA

#endif  // WAVEFUNCTION_H_

#ifndef WAVEFUNCTION_H_
#define WAVEFUNCTION_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif 
#include <vector>
#include <array>
#include "movec_parser.h"

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
  typedef Container<double, Allocator<double>> vector_double;
  typedef Container<Point, Allocator<Point>> vector_Point;

 public:
  Wavefunction() {}
  Wavefunction(vector_Point* p, const std::shared_ptr<Movec_Parser>);

  ~Wavefunction() { destroy_handle(); }

  void ao_to_mo(const vector_double&);

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
  size_t n_basis_functions;


  size_t electrons;
  size_t lda;

  vector_double psi;
  vector_double psiTau;
  vector_double movecs;
  vector_Point* pos;

 private:
  static double* get_raw_pointer(vector_double&);
  static const double* get_raw_pointer(const vector_double&);

  std::shared_ptr<void> create_handle();
  void destroy_handle();

  std::shared_ptr<void> v_handle;
};

template <> void Wavefunction<std::vector, std::allocator>::ao_to_mo(const vector_double&);
template <> double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(vector_double&);
template <> const double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(const vector_double&);
template class Wavefunction<std::vector, std::allocator>;
typedef Wavefunction<std::vector, std::allocator> Wavefunction_Host;

#ifdef HAVE_CUDA
template <> std::shared_ptr<void> Wavefunction<thrust::device_vector, thrust::device_allocator>::create_handle();
template <> void Wavefunction<thrust::device_vector, thrust::device_allocator>::destroy_handle();
template <> void Wavefunction<thrust::device_vector, thrust::device_allocator>::ao_to_mo(const vector_double&);
template <> double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(vector_double&);
template <> const double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(const vector_double&);
template class Wavefunction<thrust::device_vector, thrust::device_allocator>;
typedef Wavefunction<thrust::device_vector, thrust::device_allocator> Wavefunction_Device;
#endif  // HAVE_CUDA

#endif  // WAVEFUNCTION_H_

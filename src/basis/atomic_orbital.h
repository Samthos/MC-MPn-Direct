#ifndef ATOMIC_ORBITAL_H_
#define ATOMIC_ORBITAL_H_

#include <array>

#ifdef HAVE_CUDA
#define HOSTDEVICE __host__ __device__
#else 
#define HOSTDEVICE
#endif

class Atomic_Orbital {
 public:
  int contraction_begin;
  int contraction_end;
  int contraction_index;
  int ao_index;
  int angular_momentum;
  double pos[3];

  HOSTDEVICE void evaluate_contraction(const std::array<double, 3>&, double*, double*, double*);
  HOSTDEVICE void evaluate_contraction_with_derivative(const std::array<double, 3>&, double*, double*, double*, double*);

 private:

  HOSTDEVICE double calculate_r2(const std::array<double, 3>&);
};

#undef HOSTDEVICE

#endif  // ATOMIC_ORBITAL_H_

#include "atomic_orbital.h"

#include <cmath>
#include <iostream>

#ifdef HAVE_CUDA
#define HOSTDEVICE __host__ __device__
#else 
#define HOSTDEVICE
#endif


HOSTDEVICE 
void Atomic_Orbital::evaluate_contraction(const std::array<double, 3>& walker_pos,
    double* contraction_amplitudes,
    double* contraction_exponent,
    double* contraction_coeficient) {
  double r2 = calculate_r2(walker_pos);
  for (auto i = contraction_begin; i < contraction_end; i++) {
    contraction_amplitudes[contraction_index] += exp(-contraction_exponent[i] * r2) * contraction_coeficient[i];
  }
}

HOSTDEVICE 
void Atomic_Orbital::evaluate_contraction_with_derivative(const std::array<double, 3>& walker_pos,
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    double* contraction_exponent,
    double* contraction_coeficient) {
  double r2 = calculate_r2(walker_pos);
  for (auto i = contraction_begin; i < contraction_end; i++) {
    double alpha = contraction_exponent[i];
    double exponential = exp(-alpha * r2) * contraction_coeficient[i];
    contraction_amplitudes[contraction_index] += exponential;
    contraction_amplitudes_derivative[contraction_index] -= 2.0 * alpha * exponential;
  }
}

HOSTDEVICE
double Atomic_Orbital::calculate_r2(const std::array<double, 3>& walker_pos) {
  double r2 = 0.0;
  r2 += (pos[0] - walker_pos[0]) * (pos[0] - walker_pos[0]);
  r2 += (pos[1] - walker_pos[1]) * (pos[1] - walker_pos[1]);
  r2 += (pos[2] - walker_pos[2]) * (pos[2] - walker_pos[2]);
  return r2;
}
#undef HOSTDEVICE

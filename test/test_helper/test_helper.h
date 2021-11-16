#ifndef TEST_HELPER_H_
#define TEST_HELPER_H_

#include <thrust/device_vector.h>

#include <vector>

void vfill(std::vector<double>& v);
std::vector<double> get_vector(std::vector<double> v);
std::vector<double> make_psi(int n_electron_pairs, int n_orbitals, double sign);
double PolyGamma(int n, int z);
double PolyGamma_Difference(int start, int stop, int n);

#ifdef HAVE_CUDA
void vfill(thrust::device_vector<double>& v);
std::vector<double> get_vector(thrust::device_vector<double>& v);
#endif
#endif  // TEST_HELPER_H_

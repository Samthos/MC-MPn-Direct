#ifndef OVPS_SET_H_
#define OVPS_SET_H_

#include <vector>
#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif // HAVE_CUDA

template <class Container>
class OVPS_SET_BASE {
 public:
  OVPS_SET_BASE() = default;
  OVPS_SET_BASE(int mc_pair_num_);
  void resize(int mc_pair_num_);
  void update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda);

  int mc_pair_num;
  Container s_11, s_12, s_21, s_22;
};

template class OVPS_SET_BASE<std::vector<double>>;
typedef OVPS_SET_BASE<std::vector<double>> OVPS_SET;

#ifdef HAVE_CUDA
template class OVPS_SET_BASE<thrust::device_vector<double>>;
typedef OVPS_SET_BASE<thrust::device_vector<double>> OVPS_SET_DEVICE;

template <class T, class S>
void copy_OVPS_SET_BASE(OVPS_SET_BASE<T>& src, OVPS_SET_BASE<S>& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}
#endif  // HAVE_CUDA

#endif  // OVPS_SET_H_

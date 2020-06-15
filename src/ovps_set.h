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

void copy_OVPS_HOST_TO_DEVICE(OVPS_SET& src, OVPS_SET_DEVICE& dest);
void copy_OVPS_DEVICE_TO_HOST(OVPS_SET_DEVICE& src, OVPS_SET& dest);
#endif  // HAVE_CUDA

#endif  // OVPS_SET_H_

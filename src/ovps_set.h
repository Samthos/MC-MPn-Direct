#ifndef OVPS_Set_H_
#define OVPS_Set_H_

#include <vector>
#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif // HAVE_CUDA

template <class Container>
class OVPS_Set {
 public:
  OVPS_Set() = default;
  OVPS_Set(int mc_pair_num_);
  void resize(int mc_pair_num_);
  void update(Container& psi1Tau, int psi1_offset, Container& psi2Tau, int psi2_offset, size_t inner, size_t lda);

  int mc_pair_num;
  Container s_11, s_12, s_21, s_22;
};

template <> void OVPS_Set<std::vector<double>>::update(std::vector<double>& psi1Tau, int psi1_offset, std::vector<double>& psi2Tau, int psi2_offset, size_t inner, size_t lda);
template class OVPS_Set<std::vector<double>>;
typedef OVPS_Set<std::vector<double>> OVPS_Set_Host;

#ifdef HAVE_CUDA
template <> void OVPS_Set<thrust::device_vector<double>>::update(thrust::device_vector<double>& psi1Tau, int psi1_offset, thrust::device_vector<double>& psi2Tau, int psi2_offset, size_t inner, size_t lda);
template class OVPS_Set<thrust::device_vector<double>>;
typedef OVPS_Set<thrust::device_vector<double>> OVPS_Set_Device;

template <class T, class S>
void copy_OVPS_Set(OVPS_Set<T>& src, OVPS_Set<S>& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}
#endif  // HAVE_CUDA

#endif  // OVPS_Set_H_

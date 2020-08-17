#ifndef OVPS_Set_H_
#define OVPS_Set_H_

#include <vector>
#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif // HAVE_CUDA

template <template <class, class> class Container, template <class> class Allocator>
class OVPS_Set {
  typedef Container<double, Allocator<double>> vector_double;

 public:
  OVPS_Set() = default;
  OVPS_Set(int mc_pair_num_);
  void resize(int mc_pair_num_);
  void update(vector_double& psi1Tau, int psi1_offset, vector_double& psi2Tau, int psi2_offset, size_t inner, size_t lda);

  int mc_pair_num;
  vector_double s_11, s_12, s_21, s_22;
};

template <> 
void OVPS_Set<std::vector, std::allocator>::update(
    vector_double& psi1Tau, int psi1_offset,
    vector_double& psi2Tau, int psi2_offset, 
    size_t inner, size_t lda);
template class OVPS_Set<std::vector, std::allocator>;
typedef OVPS_Set<std::vector, std::allocator> OVPS_Set_Host;

#ifdef HAVE_CUDA
template <> void OVPS_Set<thrust::device_vector, thrust::device_allocator>::update(
    vector_double& psi1Tau, int psi1_offset,
    vector_double& psi2Tau, int psi2_offset, 
    size_t inner, size_t lda);
template class OVPS_Set<thrust::device_vector, thrust::device_allocator>;
typedef OVPS_Set<thrust::device_vector, thrust::device_allocator> OVPS_Set_Device;

void copy_OVPS_Set(OVPS_Set_Host& src, OVPS_Set_Device& dest);
void copy_OVPS_Set(OVPS_Set_Device& src, OVPS_Set_Host& dest);
#endif  // HAVE_CUDA

#endif  // OVPS_Set_H_

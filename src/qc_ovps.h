#ifndef QC_OVPS_H_
#define QC_OVPS_H_
#include <vector>

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "basis/wavefunction.h"
#include "tau.h"
#include "ovps_set.h"

template <template <class, class> class Container, template <class> class Allocator>
class OVPS {
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;
  typedef OVPS_Set<Container, Allocator> OVPS_Set_Type;

 public:
  void init(int dimm, int electron_pairs_);
  void update(Wavefunction_Type&, Wavefunction_Type&, Tau*);

  std::vector<std::vector<OVPS_Set_Type>> o_set, v_set;

 private:
  int electron_pairs;
};

template <> 
void OVPS<std::vector, std::allocator>::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau);

template class OVPS<std::vector, std::allocator>;
typedef OVPS<std::vector, std::allocator> OVPS_Host;

#ifdef HAVE_CUDA
template <> void OVPS<thrust::device_vector, thrust::device_allocator>::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau);
template class OVPS<thrust::device_vector, thrust::device_allocator>;
typedef OVPS<thrust::device_vector, thrust::device_allocator> OVPS_Device;

void copy_OVPS(OVPS_Host& src, OVPS_Device& dest);
void copy_OVPS(OVPS_Device& src, OVPS_Host& dest);
#endif

#endif  // QC_OVPS_H_

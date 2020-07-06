#ifndef QC_OVPS_H_
#define QC_OVPS_H_
#include <vector>

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "basis/wavefunction.h"
#include "tau.h"
#include "ovps_set.h"

template <class Container>
class OVPS {
 public:
  void init(int dimm, int electron_pairs_);
  void update(Wavefunction&, Wavefunction&, Tau*);

  std::vector<std::vector<OVPS_Set<Container>>> o_set, v_set;

 private:
  int electron_pairs;
};

template class OVPS<std::vector<double>>;
typedef OVPS<std::vector<double>> OVPS_Host;

#ifdef HAVE_CUDA
template class OVPS<thrust::device_vector<double>>;
typedef OVPS<thrust::device_vector<double>> OVPS_Device;

template <class T, class S>
void copy_OVPS(OVPS<T>& src, OVPS<S>& dest) {
  for (int i = 0; i < src.o_set.size(); i++) {
    for (int j = 0; j < src.o_set[i].size(); j++) {
      copy_OVPS_Set(src.o_set[i][j], dest.o_set[i][j]);
      copy_OVPS_Set(src.v_set[i][j], dest.v_set[i][j]);
    }
  }
}
#endif

#endif  // QC_OVPS_H_

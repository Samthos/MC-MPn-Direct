#ifndef QC_OVPS_H_
#define QC_OVPS_H_
#include <vector>

#include "basis/qc_basis.h"
#include "basis/nw_vectors.h"
#include "qc_random.h"
#include "tau_integrals.h"
#include "ovps_set.h"

class OVPs {
 public:
  void init(int dimm, int mc_pair_num_);
  void update_ovps(Wavefunction&, Wavefunction&, Tau*);

  std::vector<std::vector<OVPS_SET>> o_set, v_set;

 private:
  int numBand, offBand, numDiff, numBlock;
  int mc_pair_num, my_iocc1, my_iocc2, my_ivir1, my_ivir2;
  bool full;
};
#endif  // QC_OVPS_H_

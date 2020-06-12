#ifndef QC_OVPS_H_
#define QC_OVPS_H_
#include <vector>

#include "basis/qc_basis.h"
#include "basis/nw_vectors.h"
#include "qc_random.h"
#include "tau_integrals.h"
#include "ovps_set.h"



struct OVPS_ARRAY {
  double *ent, *enCore;
  std::vector<std::vector<double*>> enBlock;
  std::vector<double*> enEx1;
  std::vector<std::vector<double*>> enCov;

  double *enGrouped;

  double *en2mCore, *en2pCore;
  double *en2m, *en2p;

  double *en3_1pCore, *en3_2pCore, *en3_12pCore;
  double *en3_1mCore, *en3_2mCore, *en3_12mCore;
  double *en3_12cCore, *en3_22cCore;
  double *one, *en3c12, *en3c22;

  double *en3_1p, *en3_2p, *en3_12p;
  double *en3_1m, *en3_2m, *en3_12m;
  double *en3_c;

  double *rv;
};

class OVPs {
 public:
  void init(int dimm, int mc_pair_num_);
  void free();
  void update_ovps(Wavefunction&, Wavefunction&, Tau*);

  void init_02(int, int, int, int, int, const NWChem_Movec_Parser&, bool);
  void alloc_02();
  void free_tau_02();
  void free_02();
  void zero_energy_arrays_02();
  void update_ovps_02(const BasisData&);

  void init_03(int, int, int, int, int, const NWChem_Movec_Parser&, bool);
  void alloc_03();
  void free_tau_03();
  void free_03();
  void zero_energy_arrays_03();
  void update_ovps_03(Electron_Pair*, Tau*);

  std::vector<std::vector<OVPS_SET>> o_set, v_set;
  OVPS_ARRAY ovps, d_ovps;

 private:
  int numBand, offBand, numDiff, numBlock;
  int mc_pair_num, my_iocc1, my_iocc2, my_ivir1, my_ivir2;
  bool full;
};
#endif  // QC_OVPS_H_

// copyright 2017

#include <vector>

#include "el_pair.h"
#include "qc_basis.h"
#include "qc_random.h"

#ifndef QC_OVPS_H_
#define QC_OVPS_H_
struct OVPS_ARRAY {
  double *psi1, *psi2;
  double *occ1, *occ2, *vir1, *vir2;
  double *occTau1, *occTau2, *virTau1, *virTau2;

  double *t_val1, *tg_val1, *tgc_val1;
  double *t_save_val1, *tg_save_val1, *tgc_save_val1;

  double *os_13, *os_14, *os_23, *os_24;
  double *vs_13, *vs_14, *vs_23, *vs_24;
  double *ps_24;

  double *t_val2, *tg_val2, *tgc_val2;
  double *t_val12, *tg_val12, *tgc_val12;
  double *t_save_val2, *tg_save_val2, *tgc_save_val2;
  double *t_save_val12, *tg_save_val12, *tgc_save_val12;

  double *os_35, *os_36, *os_45, *os_46;
  double *os_15, *os_16, *os_25, *os_26;
  double *vs_35, *vs_36, *vs_45, *vs_46;
  double *vs_15, *vs_16, *vs_25, *vs_26;
  double *ps_12c, *ps_22c;

  double *ent;
  std::vector<std::vector<std::vector<double *>>> en2Block;
  std::vector<std::vector<std::vector<double *>>> en2Ex1;
  std::vector<std::vector<std::vector<double *>>> en2Ex2;
  std::vector<std::vector<double *>> en2;
  double *en2mCore, *en2pCore;
  double *en2m, *en2p;

  std::vector<std::vector<std::vector<double *>>> en3Block;
  std::vector<std::vector<std::vector<double *>>> en3Ex1;
  std::vector<std::vector<std::vector<double *>>> en3Ex2;
  std::vector<std::vector<double *>> en3;
  double *en3_1pCore, *en3_2pCore, *en3_12pCore;
  double *en3_1mCore, *en3_2mCore, *en3_12mCore;
  double *en3_12cCore, *en3_22cCore;
  double *en3_1p, *en3_2p, *en3_12p;
  double *en3_1m, *en3_2m, *en3_12m;
  double *en3_c;
  double *one, *en3c12, *en3c22;

  double *rv;
};

class OVPs {
 private:
  int numBand, offBand, numDiff, numBlock;
  int mc_pair_num, iocc1, iocc2, ivir1, ivir2;
  double lambda;

 public:
  void init_02(int, int, int, int, int, const Basis &);
  void alloc_02();
  void free_tau_02();
  void free_02();
  void zero_energy_arrays_02();
  void new_tau_02(Basis &, Random &);
  void set_tau_02(int);
  void init_tau_02(Basis &);
  void update_ovps_02(el_pair_typ *);

  void init_03(int, int, int, int, int, const Basis &);
  void alloc_03();
  void free_tau_03();
  void free_03();
  void zero_energy_arrays_03();
  void new_tau_03(Basis &, Random &);
  void set_tau_03(int, int);
  void init_tau_03(Basis &);
  void update_ovps_03(el_pair_typ *);

  OVPS_ARRAY ovps, d_ovps;

  double xx1, t1_twgt;
  double xx2, t2_twgt;
};
#endif  // QC_OVPS_H_

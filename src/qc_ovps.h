// copyright 2017

#ifndef QC_OVPS_H_
#define QC_OVPS_H_
#include <vector>

#include "basis/qc_basis.h"
#include "qc_random.h"
#include "tau_integrals.h"

#include "cblas.h"
#include "blas_calls.h"

class OVPS_SET {
 public:
  OVPS_SET() = default;
  OVPS_SET(int mc_pair_num_) {
    resize(mc_pair_num_);
  }
  void resize(int mc_pair_num_) {
    mc_pair_num = mc_pair_num_;
    s_11.resize(mc_pair_num * mc_pair_num);
    s_12.resize(mc_pair_num * mc_pair_num);
    s_21.resize(mc_pair_num * mc_pair_num);
    s_22.resize(mc_pair_num * mc_pair_num);
  }
  void update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda) {
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
        mc_pair_num, inner,
        alpha,
        psi1Tau, lda,
        beta,
        s_11.data(), mc_pair_num);
    set_Upper_from_Lower(s_11.data(), mc_pair_num);
    cblas_dscal(mc_pair_num, 0.0, s_11.data(), mc_pair_num+1);

    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
        mc_pair_num, inner,
        alpha,
        psi2Tau, lda,
        beta,
        s_22.data(), mc_pair_num);
    set_Upper_from_Lower(s_22.data(), mc_pair_num);
    cblas_dscal(mc_pair_num, 0.0, s_22.data(), mc_pair_num+1);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        mc_pair_num, mc_pair_num, inner,
        alpha,
        psi1Tau, lda,
        psi2Tau, lda,
        beta,
        s_21.data(), mc_pair_num);
    cblas_dscal(mc_pair_num, 0.0, s_21.data(), mc_pair_num+1);
    Transpose(s_21.data(), mc_pair_num, s_12.data());
  }

  int mc_pair_num;
  std::vector<double> s_11, s_12, s_21, s_22;
};

struct OVPS_ARRAY {
  double *ent, *enCore;
  std::vector<std::vector<double*>> enBlock;
  std::vector<double*> enEx1;
  std::vector<std::vector<double*>> enCov;

  double *en2mCore, *en2pCore;
  double *en3_1pCore, *en3_2pCore, *en3_12pCore;
  double *en3_1mCore, *en3_2mCore, *en3_12mCore;
  double *en3_12cCore, *en3_22cCore;
  double *one, *en3c12, *en3c22;

  double *enGrouped;
  double *en2m, *en2p;
  double *en3_1p, *en3_2p, *en3_12p;
  double *en3_1m, *en3_2m, *en3_12m;
  double *en3_c;

  double *rv;
};

class OVPs {
 public:
  void init(int dimm, int mc_pair_num_, const Basis &basis);
  void free();
  void update_ovps(Wavefunction&, Wavefunction&, Tau*);

  void init_02(int, int, int, int, int, const Basis &, bool);
  void alloc_02();
  void free_tau_02();
  void free_02();
  void zero_energy_arrays_02();
  void update_ovps_02(const BasisData&);

  void init_03(int, int, int, int, int, const Basis &, bool);
  void alloc_03();
  void free_tau_03();
  void free_03();
  void zero_energy_arrays_03();
  void update_ovps_03(Electron_Pair*, Tau*);

  std::vector<std::vector<OVPS_SET>> o_set, v_set;
  OVPS_ARRAY ovps, d_ovps;

 private:
  int numBand, offBand, numDiff, numBlock;
  int mc_pair_num, iocc1, iocc2, ivir1, ivir2;
  bool full;
};
#endif  // QC_OVPS_H_

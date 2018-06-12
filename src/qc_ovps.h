// copyright 2017

#include <vector>

#include "el_pair.h"
#include "basis/qc_basis.h"
#include "qc_random.h"
#include "blas_calls.h"
#include "tau_integrals.h"

#ifndef QC_OVPS_H_
#define QC_OVPS_H_
class OVPS_SET {
 public:
  OVPS_SET() {
    cpu_allocated = false;
    gpu_allocated = false;
  }
  OVPS_SET(int mc_pair_num_, int n1_, int n2_) :
      mc_pair_num(mc_pair_num_), n1(n1_), n2(n2_)
  {
    cpu_allocated = false;
    gpu_allocated = false;
  }
  ~OVPS_SET() {
    free();
  }
  void alloc_cpu() {
    if (cpu_allocated || gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else {
      s_11 = new double[mc_pair_num * mc_pair_num];
      s_12 = new double[mc_pair_num * mc_pair_num];
      s_21 = new double[mc_pair_num * mc_pair_num];
      s_22 = new double[mc_pair_num * mc_pair_num];
      cpu_allocated = true;
    }
  }
  void alloc_gpu() {
    if (cpu_allocated || gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else {
      throw std::runtime_error("OVPS_SET.alloc_gpu not implemented");
    }
  }

  void alloc_cpu(int mc_pair_num_, int n1_, int n2_) {
    mc_pair_num = mc_pair_num_;
    n1 = n1_;
    n2 = n2_;
    if (cpu_allocated || gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else {
      alloc_cpu();
    }
  }
  void alloc_gpu(int mc_pair_num_, int n1_, int n2_) {
    if (cpu_allocated || gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else {
      alloc_gpu();
    }
  }

  void free() {
    if (gpu_allocated) {
      free_gpu();
    } else if (cpu_allocated) {
      free_cpu();
    }
  }
  void free_cpu() {
    if (gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else if (cpu_allocated){
      delete[] s_11;
      delete[] s_12;
      delete[] s_21;
      delete[] s_22;
      cpu_allocated = false;
    }
  }
  void free_gpu() {
    if (gpu_allocated) {
      throw std::runtime_error("OVPS_SET already allocated");
    } else if (cpu_allocated){
      throw std::runtime_error("OVPS_SET.alloc_gpu not implemented");
    }
  }

  void update(double *psi1, double *psi2, double *psi1Tau, double *psi2Tau) {
    if (cpu_allocated) {
      host_update(psi1, psi2, psi1Tau, psi2Tau);
    } else if (gpu_allocated) {
      device_update(psi1, psi2, psi1Tau, psi2Tau);
    }
  }
  void host_update(double *psi1, double *psi2, double *psi1Tau, double *psi2Tau) {
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans,
                    mc_pair_num, mc_pair_num, n2 - n1,
                    alpha, psi1Tau, mc_pair_num,
                    psi1, mc_pair_num,
                    beta, s_11, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans,
                    mc_pair_num, mc_pair_num, n2 - n1,
                    alpha, psi2Tau, mc_pair_num,
                    psi2, mc_pair_num,
                    beta, s_22, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                mc_pair_num, mc_pair_num, n2 - n1,
                alpha, psi1Tau, mc_pair_num,
                psi2, mc_pair_num,
                beta, s_21, mc_pair_num);
    Transpose(s_21, mc_pair_num, s_12);
  }
  void device_update(double *psi1, double *psi2, double *psi1Tau, double *psi2Tau) {
    throw std::runtime_error("OVPS_SET.device_update not yet implemented");
  }

  int mc_pair_num, n1, n2;
  bool cpu_allocated, gpu_allocated;
  double *s_11, *s_12, *s_21, *s_22;
};

struct OVPS_ARRAY {
  double *psi1, *psi2;
  double *occ1, *occ2, *vir1, *vir2;
  double *occTau1, *occTau2, *virTau1, *virTau2;

  double *os_13, *os_14, *os_23, *os_24;
  double *vs_13, *vs_14, *vs_23, *vs_24;
  double *ps_24;

  double *os_35, *os_36, *os_45, *os_46;
  double *os_15, *os_16, *os_25, *os_26;
  double *vs_35, *vs_36, *vs_45, *vs_46;
  double *vs_15, *vs_16, *vs_25, *vs_26;
  double *ps_12c, *ps_22c;

  double *ent;
  std::vector<std::vector<double *>> en2Ex1;
  std::vector<std::vector<double *>> en2Ex2;
  std::vector<std::vector<double *>> en2;
  double *en2mCore, *en2pCore;
  double *en2m, *en2p;

  std::vector<std::vector<double *>> en3Ex1;
  std::vector<std::vector<double *>> en3Ex2;
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
  int numBand, offBand, numDiff;
  int mc_pair_num, iocc1, iocc2, ivir1, ivir2;

 public:
  void init_02(int, int, int, int, const Basis &);
  void alloc_02();
  void free_tau_02();
  void free_02();
  void zero_energy_arrays_02();
  void update_ovps_02(el_pair_typ *, Stochastic_Tau&);

  void init_03(int, int, int, int, const Basis &);
  void alloc_03();
  void free_tau_03();
  void free_03();
  void zero_energy_arrays_03();
  void update_ovps_03(el_pair_typ *, Stochastic_Tau&);

  OVPS_ARRAY ovps, d_ovps;
};
#endif  // QC_OVPS_H_

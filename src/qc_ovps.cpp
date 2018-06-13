#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "blas_calls.h"
#include "qc_ovps.h"

void print_out(double* A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%12.8f", A[i * n + j]);
    }
    printf("\n");
  }
}

void OVPs::init_02(int p1, int p2, int p3, int p4, const Basis &basis) {
  mc_pair_num = p1;
  numBand = p2;
  offBand = p3;
  numDiff = p4;

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;
}
void OVPs::alloc_02() {
  d_ovps.occ1 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.occ2 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.vir1 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.vir2 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.psi1 = new double[mc_pair_num * (ivir2 - iocc1)];
  d_ovps.psi2 = new double[mc_pair_num * (ivir2 - iocc1)];
  d_ovps.occTau1 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.occTau2 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.virTau1 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.virTau2 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.rv = new double[mc_pair_num];

  d_ovps.os_13 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_14 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_23 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_24 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_13 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_14 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_23 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_24 = new double[mc_pair_num * mc_pair_num];

  d_ovps.ps_24 = new double[mc_pair_num * mc_pair_num * numBand];

  d_ovps.en2mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en2pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en2m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en2p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];

  d_ovps.ent = new double[(ivir2 - iocc1) * mc_pair_num];

  d_ovps.en2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));

  d_ovps.en2Ex1 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en2Ex2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  for (uint32_t i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      d_ovps.en2[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      d_ovps.en2Ex1[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      d_ovps.en2Ex2[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
    }
  }
}
void OVPs::free_tau_02() {
}
void OVPs::free_02() {
  delete[] d_ovps.occ1;
  delete[] d_ovps.occ2;
  delete[] d_ovps.vir1;
  delete[] d_ovps.vir2;
  delete[] d_ovps.psi1;
  delete[] d_ovps.psi2;
  delete[] d_ovps.occTau1;
  delete[] d_ovps.occTau2;
  delete[] d_ovps.virTau1;
  delete[] d_ovps.virTau2;
  delete[] d_ovps.rv;

  delete[] d_ovps.os_13;
  delete[] d_ovps.os_14;
  delete[] d_ovps.os_23;
  delete[] d_ovps.os_24;

  delete[] d_ovps.vs_13;
  delete[] d_ovps.vs_14;
  delete[] d_ovps.vs_23;
  delete[] d_ovps.vs_24;

  delete[] d_ovps.ps_24;

  delete[] d_ovps.ent;
  delete[] d_ovps.en2mCore;
  delete[] d_ovps.en2pCore;
  delete[] d_ovps.en2m;
  delete[] d_ovps.en2p;

  for (uint32_t i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      delete[] d_ovps.en2[i][j];
      delete[] d_ovps.en2Ex1[i][j];
      delete[] d_ovps.en2Ex2[i][j];
    }
  }
}

void OVPs::init_03(int p1, int p2, int p3, int p4, const Basis &basis) {
  init_02(p1, p2, p3, p4, basis);
}
void OVPs::alloc_03() {
  alloc_02();
  d_ovps.os_15 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_16 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_25 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_26 = new double[mc_pair_num * mc_pair_num];

  d_ovps.os_35 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_36 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_45 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_46 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_15 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_16 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_25 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_26 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_35 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_36 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_45 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_46 = new double[mc_pair_num * mc_pair_num];

  d_ovps.ps_12c = new double[mc_pair_num * numBand];
  d_ovps.ps_22c = new double[mc_pair_num * numBand];

  d_ovps.en3_1pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_1mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12cCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_22cCore = new double[mc_pair_num * mc_pair_num];

  d_ovps.en3_1p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_2p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_12p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_1m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_2m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_12m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_c = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];

  d_ovps.en3c12 = new double[mc_pair_num];
  d_ovps.en3c22 = new double[mc_pair_num];
  d_ovps.one = new double[mc_pair_num];
  std::fill(d_ovps.one, d_ovps.one + mc_pair_num, 1.0);

  d_ovps.en3 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en3Ex1 =   std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en3Ex2 =   std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  for (uint32_t i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      d_ovps.en3[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      d_ovps.en3Ex1[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      d_ovps.en3Ex2[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
    }
  }
}
void OVPs::free_tau_03() {
  free_tau_02();
}
void OVPs::free_03() {
  free_02();

  delete[] d_ovps.os_15;
  delete[] d_ovps.os_16;
  delete[] d_ovps.os_25;
  delete[] d_ovps.os_26;

  delete[] d_ovps.os_35;
  delete[] d_ovps.os_36;
  delete[] d_ovps.os_45;
  delete[] d_ovps.os_46;

  delete[] d_ovps.vs_15;
  delete[] d_ovps.vs_16;
  delete[] d_ovps.vs_25;
  delete[] d_ovps.vs_26;

  delete[] d_ovps.vs_35;
  delete[] d_ovps.vs_36;
  delete[] d_ovps.vs_45;
  delete[] d_ovps.vs_46;

  delete[] d_ovps.ps_12c;
  delete[] d_ovps.ps_22c;

  delete[] d_ovps.en3_1pCore;
  delete[] d_ovps.en3_2pCore;
  delete[] d_ovps.en3_12pCore;
  delete[] d_ovps.en3_1mCore;
  delete[] d_ovps.en3_2mCore;
  delete[] d_ovps.en3_12mCore;
  delete[] d_ovps.en3_12cCore;
  delete[] d_ovps.en3_22cCore;

  delete[] d_ovps.en3_1p;
  delete[] d_ovps.en3_2p;
  delete[] d_ovps.en3_12p;
  delete[] d_ovps.en3_1m;
  delete[] d_ovps.en3_2m;
  delete[] d_ovps.en3_12m;
  delete[] d_ovps.en3_c;

  delete[] d_ovps.one;
  delete[] d_ovps.en3c12;
  delete[] d_ovps.en3c22;

  for (uint32_t i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      delete[] d_ovps.en3[i][j];
      delete[] d_ovps.en3Ex1[i][j];
      delete[] d_ovps.en3Ex2[i][j];
    }
  }
}

void OVPs::zero_energy_arrays_02() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
}
void OVPs::zero_energy_arrays_03() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
  for (auto& it : d_ovps.en3) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
}

void freq_indp_gf(OVPS_ARRAY ovps, int mc_pair_num, int iocc2, int offBand, int numBand) {
  int tidx, tidy, index, lookup;
  for (tidx = 0; tidx < mc_pair_num; tidx++) {
    for (tidy = 0; tidy < numBand; tidy++) {
      index = tidy * mc_pair_num + tidx;
      if (tidy - offBand < 0) {
        lookup = (iocc2 - offBand + tidy) * mc_pair_num + tidx;
        ovps.ps_12c[index] = ovps.occ1[lookup] * ovps.occ2[lookup];
        ovps.ps_22c[index] = ovps.occ2[lookup] * ovps.occ2[lookup];
      } else {
        lookup = (tidy - offBand) * mc_pair_num + tidx;
        ovps.ps_12c[index] = ovps.vir1[lookup] * ovps.vir2[lookup];
        ovps.ps_22c[index] = ovps.vir2[lookup] * ovps.vir2[lookup];
      }
    }
  }
}

void OVPs::update_ovps_02(el_pair_typ* el_pair_list, Stochastic_Tau& tau) {
  int ip, am;
  double alpha = 1.00;
  double beta = 0.00;

  //copy weights from el_pair_list to host arrays
  for (ip = 0; ip < mc_pair_num; ip++) {
    if (el_pair_list[ip].is_new) {
      d_ovps.rv[ip] = el_pair_list[ip].rv;
      for (am = iocc1; am < iocc2; am++) {
        d_ovps.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
        d_ovps.occ1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.occ2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
      }
      for (am = ivir1; am < ivir2; am++) {
        d_ovps.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
        d_ovps.vir1[(am - ivir1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.vir2[(am - ivir1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
      }
    }
  }
  //copy wave functions from host to device;
  {
    auto t_val = tau.get_exp_tau(0, 0);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_13, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_24, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_23, mc_pair_num);
    Transpose(d_ovps.os_23, mc_pair_num, d_ovps.os_14);

    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_13, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_24, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_23, mc_pair_num);
    Transpose(d_ovps.vs_23, mc_pair_num, d_ovps.vs_14);
  }


  std::fill(d_ovps.ps_24, d_ovps.ps_24 + mc_pair_num * mc_pair_num * numBand, 0.0);
  for (am = 0; am < numBand; am++) {
    if (am - offBand < 0) {  //construct ps_?? and ps_??c for occupied orbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num, alpha, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    } else {  //construct ps_?? and ps_??c for virtualorbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num, alpha, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    }
  }
}
void OVPs::update_ovps_03(el_pair_typ* el_pair_list, Stochastic_Tau& tau) {
  double alpha = 1.00;
  double beta = 0.00;

  update_ovps_02(el_pair_list, tau);

  freq_indp_gf(d_ovps, mc_pair_num, iocc2 - iocc1, offBand, numBand);

  {
    auto t_val = tau.get_exp_tau(1, 1);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_35, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_46, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_45, mc_pair_num);
    Transpose(d_ovps.os_45, mc_pair_num, d_ovps.os_36);

    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_35, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_46, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_45, mc_pair_num);
    Transpose(d_ovps.vs_45, mc_pair_num, d_ovps.vs_36);
  }


  {
    auto t_val = tau.get_exp_tau(0, 1);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_15, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_26, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_25, mc_pair_num);
    Transpose(d_ovps.os_25, mc_pair_num, d_ovps.os_16);

    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau2, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_15, mc_pair_num);
    cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_26, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_25, mc_pair_num);
    Transpose(d_ovps.vs_25, mc_pair_num, d_ovps.vs_16);
  }
}

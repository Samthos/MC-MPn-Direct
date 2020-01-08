#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "blas_calls.h"
#include "qc_ovps.h"

void OVPs::init(const int dimm, const int mc_pair_num_, const Basis &basis) {
  mc_pair_num = mc_pair_num_;
  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;

  o_set.resize(dimm);
  v_set.resize(dimm);
  for (auto stop = 0; stop < dimm; stop++) {
    o_set[stop].resize(stop + 1);
    v_set[stop].resize(stop + 1);
    for (auto start = 0; start < stop + 1; start++) {
      o_set[stop][start].resize(mc_pair_num, iocc2 - iocc1, ivir2 - iocc1);
      v_set[stop][start].resize(mc_pair_num, ivir2 - ivir1, ivir2 - iocc1);
    }
  }
}
void OVPs::free() {
}
void OVPs::update_ovps(BasisData& basis, Electron_Pair_List* electron_pair_list, Tau* tau) {
  // update green's function trace objects
  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), sqrt);
      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, mc_pair_num, basis.wfn_psi1->psi.data(), ivir2 - iocc1, &t_val[iocc1], 1, basis.wfn_psi1->psiTau.data(), ivir2 - iocc1);
      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, mc_pair_num, basis.wfn_psi2->psi.data(), ivir2 - iocc1, &t_val[iocc1], 1, basis.wfn_psi2->psiTau.data(), ivir2 - iocc1);
      o_set[stop][start].update(basis.wfn_psi1->occTau(), basis.wfn_psi2->occTau());
      v_set[stop][start].update(basis.wfn_psi1->virTau(), basis.wfn_psi2->virTau());
    }
  }
}

void OVPs::init_02(int p1, int p2, int p3, int p4, int p5, const Basis &basis, bool need_full) {
  mc_pair_num = p1;
  numBand = p2;
  offBand = p3;
  numDiff = p4;
  numBlock = p5;
  full = need_full;

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;
}
void OVPs::alloc_02() {
  d_ovps.en2mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en2pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.enCore = new double[mc_pair_num * mc_pair_num];

  if (!full) {
    d_ovps.ent = new double[mc_pair_num];
  } else {
    d_ovps.ent = new double[(ivir2-iocc1) * mc_pair_num];
    if (numDiff > 1) {
      d_ovps.enGrouped = new double[(ivir2-iocc1) * (ivir2-iocc1) * 2];
      d_ovps.en2p = d_ovps.enGrouped + 0 * (ivir2-iocc1) * (ivir2-iocc1);
      d_ovps.en2m = d_ovps.enGrouped + 1 * (ivir2-iocc1) * (ivir2-iocc1);
    }

    d_ovps.enEx1 = std::vector<double*>(numBand);
    for (auto &it : d_ovps.enEx1) {
      it = new double[(ivir2-iocc1) * (ivir2-iocc1) * numDiff];
      std::fill(it, it + (ivir2-iocc1) * (ivir2-iocc1) * numDiff, 0.0);
    }

    d_ovps.enCov = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numBlock));
    for (auto &it : d_ovps.enCov) {
      for (auto &jt : it) {
        jt = new double[(ivir2-iocc1) * (ivir2-iocc1) * numDiff * (numDiff + 1) / 2];
        std::fill(jt, jt + (ivir2-iocc1) * (ivir2-iocc1) * numDiff * (numDiff + 1) / 2, 0.0);
      }
    }

    d_ovps.enBlock = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numBlock));
    for (auto &it : d_ovps.enBlock) {
      for (auto &jt : it) {
        jt = new double[(ivir2 - iocc1) * (ivir2 - iocc1) * numDiff];
      }
    }
  }
}
void OVPs::free_tau_02() {
}
void OVPs::free_02() {
  delete[] d_ovps.ent;
  delete[] d_ovps.enCore;
  delete[] d_ovps.en2mCore;
  delete[] d_ovps.en2pCore;

  if (full) {
    if (numDiff > 1) {
      delete[] d_ovps.enGrouped;
      d_ovps.en2m = nullptr;
      d_ovps.en2p = nullptr;
      d_ovps.en3_1p  = nullptr;
      d_ovps.en3_2p  = nullptr;
      d_ovps.en3_12p = nullptr;
      d_ovps.en3_1m  = nullptr;
      d_ovps.en3_2m  = nullptr;
      d_ovps.en3_12m = nullptr;
      d_ovps.en3_c   = nullptr;
    }
    for (auto &it : d_ovps.enEx1) {
      delete[] it;
    }
    for (auto &it : d_ovps.enCov) {
      for (auto &jt : it) {
        delete[] jt;
      }
    }
    for (auto &it : d_ovps.enBlock) {
      for (auto &jt : it) {
        delete[] jt;
      }
    }
  }
}

void OVPs::init_03(int p1, int p2, int p3, int p4, int p5, const Basis &basis, bool need_full) {
  init_02(p1, p2, p3, p4, p5, basis, need_full);
}
void OVPs::alloc_03() {
  alloc_02();

  d_ovps.en3_1pCore  = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2pCore  = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_1mCore  = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2mCore  = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12cCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_22cCore = new double[mc_pair_num * mc_pair_num];

  if (full && numDiff > 1) {
    delete[] d_ovps.enGrouped;
    d_ovps.en2m = nullptr;
    d_ovps.en2p = nullptr;

    d_ovps.enGrouped = new double[(ivir2 - iocc1) * (ivir2 - iocc1) * 7];
    d_ovps.en2p = d_ovps.enGrouped + 0 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en2m = d_ovps.enGrouped + 3 * (ivir2 - iocc1) * (ivir2 - iocc1);

    d_ovps.en3_1p  = d_ovps.enGrouped + 0 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_2p  = d_ovps.enGrouped + 1 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_12p = d_ovps.enGrouped + 2 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_1m  = d_ovps.enGrouped + 3 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_2m  = d_ovps.enGrouped + 4 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_12m = d_ovps.enGrouped + 5 * (ivir2 - iocc1) * (ivir2 - iocc1);
    d_ovps.en3_c   = d_ovps.enGrouped + 6 * (ivir2 - iocc1) * (ivir2 - iocc1);
  }

  d_ovps.en3c12= new double[mc_pair_num];
  d_ovps.en3c22= new double[mc_pair_num];
  d_ovps.one= new double[mc_pair_num];
  std::fill(d_ovps.one, d_ovps.one+mc_pair_num, 1.0);
}
void OVPs::free_tau_03() {
  free_tau_02();
}
void OVPs::free_03() {
  free_02();

  delete[] d_ovps.en3_1pCore ;
  delete[] d_ovps.en3_2pCore ;
  delete[] d_ovps.en3_12pCore;
  delete[] d_ovps.en3_1mCore ;
  delete[] d_ovps.en3_2mCore ;
  delete[] d_ovps.en3_12mCore;
  delete[] d_ovps.en3_12cCore;
  delete[] d_ovps.en3_22cCore;

  delete[] d_ovps.one ;
  delete[] d_ovps.en3c12;
  delete[] d_ovps.en3c22;
}

void OVPs::zero_energy_arrays_02() {
  for (auto& it : d_ovps.enBlock) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
    }
  }
}
void OVPs::zero_energy_arrays_03() {
  for (auto& it : d_ovps.enBlock) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
    }
  }
}

void freq_indp_gf(OVPS_ARRAY ovps, int mc_pair_num, int iocc2, int offBand, int numBand) {
  /*
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
  */
}

void OVPs::update_ovps_02(const BasisData& basis) {
  /*
  double alpha = 1.00;

  std::fill(d_ovps.ps_24, d_ovps.ps_24 + mc_pair_num * mc_pair_num * numBand, 0.0);
  for (int am = 0; am < numBand; am++) {
    if (am - offBand < 0) {  //construct ps_?? and ps_??c for occupied orbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num,
          alpha, basis.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1,
          basis.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1,
          d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    } else {  //construct ps_?? and ps_??c for virtualorbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num,
          alpha, basis.vir2 + (am - offBand) * mc_pair_num, 1,
          basis.vir2 + (am - offBand) * mc_pair_num, 1,
          d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    }
  }
  */
}
void OVPs::update_ovps_03(Electron_Pair* electron_pair_list, Tau* tau) {
  /*
  double alpha = 1.00;
  double beta = 0.00;

  std::cerr << "broke as hell\n";
  // update_ovps_02(electron_pair_list, tau);

  freq_indp_gf(d_ovps, mc_pair_num, iocc2 - iocc1, offBand, numBand);

  {
    auto t_val = tau->get_exp_tau(1, 1);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau2, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_35, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_46, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_45, mc_pair_num);
    Transpose(d_ovps.os_45, mc_pair_num, d_ovps.os_36);

    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau2, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_35, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_46, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_45, mc_pair_num);
    Transpose(d_ovps.vs_45, mc_pair_num, d_ovps.vs_36);
  }


  {
    auto t_val = tau->get_exp_tau(1, 0);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val[iocc1], 1, d_ovps.occTau2, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_15, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_26, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_25, mc_pair_num);
    Transpose(d_ovps.os_25, mc_pair_num, d_ovps.os_16);

    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau1, mc_pair_num);
    Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val[ivir1], 1, d_ovps.virTau2, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_15, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_26, mc_pair_num);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_25, mc_pair_num);
    Transpose(d_ovps.vs_25, mc_pair_num, d_ovps.vs_16);
  }
  */
}

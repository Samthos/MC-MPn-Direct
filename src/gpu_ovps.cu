#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <cuda.h>
#include "cublas_v2.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "cublasStatus_t_getErrorString.h"
#include "qc_ovps.h"

void OVPs::init_02(int p1, int p2, int p3, int p4, const Basis &basis) {
  mc_pair_num = p1;
  numBand = p2;
  offBand = p3;
  numDiff = p4;

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;
  lambda = 2.0 * (basis.nw_en[ivir1] - basis.nw_en[iocc2 - 1]);

  cudaError_t_Assert(cudaMallocHost((void**)&ovps.rv, sizeof(double) * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMallocHost((void**)&ovps.occ1, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.occ2, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.vir1, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.vir2, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.occTau1, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.occTau2, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.virTau1, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.virTau2, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
}
void OVPs::alloc_02() {
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.rv, sizeof(double) * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.occ1, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.occ2, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vir1, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vir2, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.psi1, sizeof(double) * mc_pair_num * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.psi2, sizeof(double) * mc_pair_num * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.occTau1, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.occTau2, sizeof(double) * mc_pair_num * (iocc2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.virTau1, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.virTau2, sizeof(double) * mc_pair_num * (ivir2 - ivir1)), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_13, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_14, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_23, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_24, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_13, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_14, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_23, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_24, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.ps_24, sizeof(double) * mc_pair_num * mc_pair_num * numBand), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en2mCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en2pCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en2m, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en2p, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.ent, sizeof(double) * (ivir2 - iocc1) * mc_pair_num), __FILE__, __LINE__);

  d_ovps.en2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));

  d_ovps.en2Ex1 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en2Ex2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  for (auto i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (auto j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      cudaError_t_Assert(cudaMalloc((void**)&(d_ovps.en2[i][j]), sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

      cudaError_t_Assert(cudaMalloc((void**)&(d_ovps.en2Ex1[i][j]), sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
      cudaError_t_Assert(cudaMalloc((void**)&(d_ovps.en2Ex2[i][j]), sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

      cudaError_t_Assert(cudaMemset(d_ovps.en2Ex1[i][j], 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
      cudaError_t_Assert(cudaMemset(d_ovps.en2Ex2[i][j], 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
    }
  }
}
void OVPs::free_tau_02() {
  cudaError_t_Assert(cudaFreeHost(ovps.rv), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFreeHost(ovps.occ1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.occ2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.vir1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.vir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.occTau1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.occTau2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.virTau1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.virTau2), __FILE__, __LINE__);
}
void OVPs::free_02() {
  cudaError_t_Assert(cudaFree(d_ovps.rv), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.psi1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.psi2), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.occ1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.occ2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vir1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.occTau1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.occTau2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.virTau1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.virTau2), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.os_13), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_14), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_23), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_24), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.vs_13), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_14), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_23), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_24), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.ps_24), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.ent), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en2mCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en2pCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en2m), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en2p), __FILE__, __LINE__);

  for (auto i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (auto j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      cudaError_t_Assert(cudaFree(d_ovps.en2[i][j]), __FILE__, __LINE__);
      cudaError_t_Assert(cudaFree(d_ovps.en2Ex1[i][j]), __FILE__, __LINE__);
      cudaError_t_Assert(cudaFree(d_ovps.en2Ex2[i][j]), __FILE__, __LINE__);
    }
  }
}

void OVPs::init_03(int p1, int p2, int p3, int p4, const Basis &basis) {
  init_02(p1, p2, p3, p4, basis);
}
void OVPs::alloc_03() {
  alloc_02();
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_15, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_16, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_25, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_26, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_35, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_36, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_45, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.os_46, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_15, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_16, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_25, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_26, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_35, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_36, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_45, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.vs_46, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.ps_12c, sizeof(double) * mc_pair_num * numBand), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.ps_22c, sizeof(double) * mc_pair_num * numBand), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_1pCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_2pCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_12pCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_1mCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_2mCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_12mCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_12cCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_22cCore, sizeof(double) * mc_pair_num * mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_1p, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_2p, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_12p, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_1m, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_2m, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_12m, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3_c, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.one, sizeof(double) * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3c12, sizeof(double) * mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3c22, sizeof(double) * mc_pair_num), __FILE__, __LINE__);

  d_ovps.en3 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en3Ex1 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en3Ex2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  for (auto i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (auto j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      cudaError_t_Assert(cudaMalloc((void**)&d_ovps.en3[i][j], sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

      cudaError_t_Assert(cudaMalloc((void**)&(d_ovps.en3Ex1[i][j]), sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
      cudaError_t_Assert(cudaMalloc((void**)&(d_ovps.en3Ex2[i][j]), sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);

      cudaError_t_Assert(cudaMemset(d_ovps.en3Ex1[i][j], 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
      cudaError_t_Assert(cudaMemset(d_ovps.en3Ex2[i][j], 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
    }
  }

  std::vector<double> one(mc_pair_num);
  std::fill(one.begin(), one.end(), 1.0);
  cudaError_t_Assert(cudaMemcpy(d_ovps.one, one.data(), sizeof(double) * one.size(), cudaMemcpyHostToDevice), __FILE__, __LINE__);
}
void OVPs::free_tau_03() {
  free_tau_02();
}
void OVPs::free_03() {
  free_02();

  cudaError_t_Assert(cudaFree(d_ovps.os_15), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_16), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_25), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_26), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.os_35), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_36), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_45), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.os_46), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.vs_15), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_16), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_25), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_26), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.vs_35), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_36), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_45), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.vs_46), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.ps_12c), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.ps_22c), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.en3_1pCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_2pCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_12pCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_1mCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_2mCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_12mCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_12cCore), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_22cCore), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.en3_1p), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_2p), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_12p), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_1m), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_2m), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_12m), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3_c), __FILE__, __LINE__);

  cudaError_t_Assert(cudaFree(d_ovps.one), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3c12), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.en3c22), __FILE__, __LINE__);

  for (auto i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (auto j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      cudaError_t_Assert(cudaFree(d_ovps.en3[i][j]), __FILE__, __LINE__);
      cudaError_t_Assert(cudaFree(d_ovps.en3Ex1[i][j]), __FILE__, __LINE__);
      cudaError_t_Assert(cudaFree(d_ovps.en3Ex2[i][j]), __FILE__, __LINE__);
    }
  }
}

void OVPs::zero_energy_arrays_02() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      cudaMemset(jt, 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1));
    }
  }
}
void OVPs::zero_energy_arrays_03() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      cudaMemset(jt, 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1));
    }
  }
  for (auto& it : d_ovps.en3) {
    for (auto& jt : it) {
      cudaMemset(jt, 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1));
    }
  }
}

__global__ void freq_indp_gf(OVPS_ARRAY ovps, int mc_pair_num, int iocc2, int offBand, int numBand) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  if (tidx < mc_pair_num) {
    int index = tidy * mc_pair_num + tidx;
    if (tidy - offBand < 0) {
      int lookup = (iocc2 - offBand + tidy) * mc_pair_num + tidx;
      ovps.ps_12c[index] = ovps.occ1[lookup] * ovps.occ2[lookup];
      ovps.ps_22c[index] = ovps.occ2[lookup] * ovps.occ2[lookup];
    } else {
      int lookup = (tidy - offBand) * mc_pair_num + tidx;
      ovps.ps_12c[index] = ovps.vir1[lookup] * ovps.vir2[lookup];
      ovps.ps_22c[index] = ovps.vir2[lookup] * ovps.vir2[lookup];
    }
  }
}

__global__ void print_out(double* A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%12.8f", A[i * n + j]);
    }
    printf("\n");
  }
}
void OVPs::update_ovps_02(el_pair_typ* el_pair_list, Stochastic_Tau& tau) {
  int ip, am;
  double alpha = 1.00;
  double beta = 0.00;
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);

  //copy weights from el_pair_list to host arrays
  for (ip = 0; ip < mc_pair_num; ip++) {  //do i = 1, el_pair_num - 1
    ovps.rv[ip] = el_pair_list[ip].rv;
  }
  cudaError_t_Assert(cudaMemcpy(d_ovps.rv, ovps.rv, sizeof(double) * mc_pair_num, cudaMemcpyHostToDevice), __FILE__, __LINE__);

  auto t_val1 =  tau.get_exp_tau_device({0});

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val1[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_13, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_23, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val1[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_14, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_24, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val1[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_13, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_23, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val1[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_14, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_24, mc_pair_num), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMemset(d_ovps.ps_24, 0, sizeof(double) * mc_pair_num * mc_pair_num * numBand), __FILE__, __LINE__);
  for (am = 0; am < numBand; am++) {
    alpha = 1.00;
    if (am - offBand < 0) {  //construct ps_?? and ps_??c for occupied orbitals
      cublasStatusAssert(cublasDger(handle, mc_pair_num, mc_pair_num, &alpha, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num), __FILE__, __LINE__);
    } else {  //construct ps_?? and ps_??c for virtualorbitals
      cublasStatusAssert(cublasDger(handle, mc_pair_num, mc_pair_num, &alpha, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num), __FILE__, __LINE__);
    }
  }

  cudaError_t_Assert(cudaThreadSynchronize(), __FILE__, __LINE__);
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}
void OVPs::update_ovps_03(el_pair_typ* el_pair_list, Stochastic_Tau& tau) {
  double alpha = 1.00;
  double beta = 0.00;
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);

  update_ovps_02(el_pair_list, tau);

  //copy wave functions from host to device;
  auto t_val2 =  tau.get_exp_tau_device({1});
  auto t_val12 =  tau.get_exp_tau_device({0, 1});

  dim3 blockSize(128, 1, 1);
  dim3 gridSize((mc_pair_num + 127) / 128, numBand, 1);
  freq_indp_gf<<<gridSize, blockSize>>>(d_ovps, mc_pair_num, iocc2 - iocc1, offBand, numBand);
  cudaError_t_Assert(cudaPeekAtLastError(), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val2[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_35, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_45, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val2[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_36, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_46, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val2[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_35, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_45, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val2[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_36, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_46, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &t_val12[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_15, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_25, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &t_val12[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_16, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_26, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &t_val12[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_15, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_25, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &t_val12[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_16, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_26, mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaThreadSynchronize(), __FILE__, __LINE__);
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

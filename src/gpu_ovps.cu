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

#ifdef QUAD_TAU
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.t_save_val1, sizeof(double) * 21 * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tg_save_val1, sizeof(double) * 21 * numBand), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tgc_save_val1, sizeof(double) * 21 * numBand), __FILE__, __LINE__);
#else
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.t_val1, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tg_val1, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tgc_val1, sizeof(double) * ivir2), __FILE__, __LINE__);
#endif  // QUAD_TAU

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
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.t_val1, sizeof(double) * ivir2), __FILE__, __LINE__);

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
#ifdef QUAD_TAU
  cudaError_t_Assert(cudaFreeHost(ovps.t_save_val1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tg_save_val1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tgc_save_val1), __FILE__, __LINE__);
#else
  cudaError_t_Assert(cudaFreeHost(ovps.t_val1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tg_val1), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tgc_val1), __FILE__, __LINE__);
#endif  // QUAD_TAU

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
  cudaError_t_Assert(cudaFree(d_ovps.t_val1), __FILE__, __LINE__);
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

#ifdef QUAD_TAU
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.t_save_val2, sizeof(double) * 21 * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tg_save_val2, sizeof(double) * 21 * numBand), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tgc_save_val2, sizeof(double) * 21 * numBand), __FILE__, __LINE__);
#else
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.t_val2, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tg_val2, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tgc_val2, sizeof(double) * ivir2), __FILE__, __LINE__);
#endif  // QUAD_TAU
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.t_val12, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tg_val12, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMallocHost((void**)&ovps.tgc_val12, sizeof(double) * ivir2), __FILE__, __LINE__);
}
void OVPs::alloc_03() {
  alloc_02();
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.t_val2, sizeof(double) * ivir2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void**)&d_ovps.t_val12, sizeof(double) * ivir2), __FILE__, __LINE__);

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

#ifdef QUAD_TAU
  cudaError_t_Assert(cudaFreeHost(ovps.t_save_val2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tg_save_val2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tgc_save_val2), __FILE__, __LINE__);
#else
  cudaError_t_Assert(cudaFreeHost(ovps.t_val2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tg_val2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tgc_val2), __FILE__, __LINE__);
#endif  // QUAD_TAU
  cudaError_t_Assert(cudaFreeHost(ovps.t_val12), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tg_val12), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFreeHost(ovps.tgc_val12), __FILE__, __LINE__);
}
void OVPs::free_03() {
  free_02();

  cudaError_t_Assert(cudaFree(d_ovps.t_val2), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_ovps.t_val12), __FILE__, __LINE__);

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

void OVPs::new_tau_02(Basis& basis, Random& random) {
  int im, am;
  double en_i, en_a;

  double p = random.get_rand();
  xx1 = -log(1.0 - p) / lambda;
  t1_twgt = 1.0 / (lambda * (1.0 - p));

  p = random.get_rand();
  xx2 = -log(1.0 - p) / lambda;
  t2_twgt = t1_twgt / (lambda * (1.0 - p));

  for (im = iocc1; im < iocc2; im++) {
    en_i = basis.nw_en[im];
    ovps.t_val1[im] = exp(en_i * xx1);
  }
  for (am = ivir1; am < ivir2; am++) {
    en_a = basis.nw_en[am];
    ovps.t_val1[am] = exp(-en_a * xx1);
  }
  for (am = 0; am < numBand; ++am) {
    en_a = basis.nw_en[iocc2 - offBand + am];
    ovps.tg_val1[am] = exp(en_a * xx1);
    ovps.tgc_val1[am] = exp(-en_a * xx1);
  }
}
void OVPs::new_tau_03(Basis& basis, Random& random) {
  int im, am;
  double en_i, en_a;

  double p = random.get_rand();
  xx1 = -log(1.0 - p) / lambda;
  t1_twgt = 1.0 / (lambda * (1.0 - p));

  p = random.get_rand();
  xx2 = -log(1.0 - p) / lambda;
  t2_twgt = t1_twgt / (lambda * (1.0 - p));

  for (im = iocc1; im < iocc2; im++) {
    en_i = basis.nw_en[im];
    ovps.t_val1[im] = exp(en_i * xx1);

    ovps.t_val2[im] = exp(en_i * xx2);
    ovps.t_val12[im] = ovps.t_val1[im] * ovps.t_val2[im];
  }
  for (am = ivir1; am < ivir2; am++) {
    en_a = basis.nw_en[am];
    ovps.t_val1[am] = exp(-en_a * xx1);

    ovps.t_val2[am] = exp(-en_a * xx2);
    ovps.t_val12[am] = ovps.t_val1[am] * ovps.t_val2[am];
  }
  for (am = 0; am < numBand; ++am) {
    en_a = basis.nw_en[iocc2 - offBand + am];
    ovps.tg_val1[am] = exp(en_a * xx1);
    ovps.tgc_val1[am] = exp(-en_a * xx1);

    ovps.tg_val2[am] = exp(en_a * xx2);
    ovps.tgc_val2[am] = exp(-en_a * xx2);
    ovps.tg_val12[am] = ovps.tg_val1[am] * ovps.tg_val2[am];
    ovps.tgc_val12[am] = ovps.tgc_val1[am] * ovps.tgc_val2[am];
  }
}

void OVPs::set_tau_02(int t1) {
  static std::array<double, 21> tauWgt = {
      1240.137264162088286,
      0.005872796340197,
      95.637046659066982,
      0.016712318843111,
      22.450252490071218,
      0.029395130776845,
      8.242542564370559,
      0.043145326054643,
      3.876926315587520,
      0.058729927034166,
      2.128605721895366,
      0.077568088880637,
      1.291883267060868,
      0.101131338617294,
      0.839203153977961,
      0.131127958307208,
      0.573533895185722,
      0.170432651403897,
      0.407885334132820,
      0.223862010922047,
      0.298891108005834};
  static std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};
  ovps.t_val1 = ovps.t_save_val1 + t1 * ivir2;
  ovps.tg_val1 = ovps.tg_save_val1 + t1 * numBand;
  ovps.tgc_val1 = ovps.tgc_save_val1 + t1 * numBand;

  t1_twgt = tauWgt[t1];
  xx1 = xx[t1];
}
void OVPs::set_tau_03(int t1, int t2) {
  static std::array<double, 21> tauWgt = {
      1240.137264162088286,
      0.005872796340197,
      95.637046659066982,
      0.016712318843111,
      22.450252490071218,
      0.029395130776845,
      8.242542564370559,
      0.043145326054643,
      3.876926315587520,
      0.058729927034166,
      2.128605721895366,
      0.077568088880637,
      1.291883267060868,
      0.101131338617294,
      0.839203153977961,
      0.131127958307208,
      0.573533895185722,
      0.170432651403897,
      0.407885334132820,
      0.223862010922047,
      0.298891108005834};
  static std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};
  ovps.t_val1 = ovps.t_save_val1 + t1 * ivir2;
  ovps.tg_val1 = ovps.tg_save_val1 + t1 * numBand;
  ovps.tgc_val1 = ovps.tgc_save_val1 + t1 * numBand;

  t1_twgt = tauWgt[t1];
  xx1 = xx[t1];

  ovps.t_val2 = ovps.t_save_val2 + t2 * ivir2;
  ovps.tg_val2 = ovps.tg_save_val2 + t2 * numBand;
  ovps.tgc_val2 = ovps.tgc_save_val2 + t2 * numBand;

  std::transform(ovps.t_val1, ovps.t_val1 + ivir2, ovps.t_val2, ovps.t_val12, std::multiplies<double>());
  std::transform(ovps.tg_val1, ovps.tg_val1 + numBand, ovps.tg_val2, ovps.tg_val12, std::multiplies<double>());
  std::transform(ovps.tgc_val1, ovps.tgc_val1 + numBand, ovps.tgc_val2, ovps.tgc_val12, std::multiplies<double>());

  t2_twgt = tauWgt[t1] * tauWgt[t2];
  xx2 = xx[t1];
}

void OVPs::init_tau_02(Basis& basis) {
  std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};

  for (uint it = 0; it < xx.size(); it++) {
    for (int jt = 0; jt < iocc2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(en * xx[it]);
    }
    for (int jt = ivir1; jt < ivir2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(-en * xx[it]);
    }
    for (int jt = 0; jt < numBand; ++jt) {
      double en = basis.nw_en[iocc2 - offBand + jt];
      ovps.tg_save_val1[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val1[it * numBand + jt] = exp(-en * xx[it]);
    }
  }
}
void OVPs::init_tau_03(Basis& basis) {
  std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};

  for (uint it = 0; it < xx.size(); it++) {
    for (int jt = 0; jt < iocc2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(en * xx[it]);
      ovps.t_save_val2[it * ivir2 + jt] = exp(en * xx[it]);
    }
    for (int jt = ivir1; jt < ivir2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(-en * xx[it]);
      ovps.t_save_val2[it * ivir2 + jt] = exp(-en * xx[it]);
    }
    for (int jt = 0; jt < numBand; ++jt) {
      double en = basis.nw_en[iocc2 - offBand + jt];
      ovps.tg_save_val1[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val1[it * numBand + jt] = exp(-en * xx[it]);
      ovps.tg_save_val2[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val2[it * numBand + jt] = exp(-en * xx[it]);
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
void OVPs::update_ovps_02(el_pair_typ* el_pair_list) {
  int ip, am;
  double alpha = 1.00;
  double beta = 0.00;
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);

  //copy weights from el_pair_list to host arrays
  for (ip = 0; ip < mc_pair_num; ip++) {  //do i = 1, el_pair_num - 1
    ovps.rv[ip] = el_pair_list[ip].rv;
  }

  //copy wave functions from host to device;
  cudaError_t_Assert(cudaMemcpy(d_ovps.rv, ovps.rv, sizeof(double) * mc_pair_num, cudaMemcpyHostToDevice), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(d_ovps.t_val1, ovps.t_val1, sizeof(double) * ivir2, cudaMemcpyHostToDevice), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &d_ovps.t_val1[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_13, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_23, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &d_ovps.t_val1[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_14, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_24, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &d_ovps.t_val1[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_13, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_23, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &d_ovps.t_val1[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
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
void OVPs::update_ovps_03(el_pair_typ* el_pair_list) {
  double alpha = 1.00;
  double beta = 0.00;
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);

  update_ovps_02(el_pair_list);

  //copy wave functions from host to device;
  cudaError_t_Assert(cudaMemcpy(d_ovps.t_val2, ovps.t_val2, sizeof(double) * ivir2, cudaMemcpyHostToDevice), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(d_ovps.t_val12, ovps.t_val12, sizeof(double) * ivir2, cudaMemcpyHostToDevice), __FILE__, __LINE__);

  dim3 blockSize(128, 1, 1);
  dim3 gridSize((mc_pair_num + 127) / 128, numBand, 1);
  freq_indp_gf<<<gridSize, blockSize>>>(d_ovps, mc_pair_num, iocc2 - iocc1, offBand, numBand);
  cudaError_t_Assert(cudaPeekAtLastError(), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &d_ovps.t_val2[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_35, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_45, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &d_ovps.t_val2[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_36, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_46, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &d_ovps.t_val2[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_35, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_45, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &d_ovps.t_val2[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_36, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_46, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &d_ovps.t_val12[iocc1], 1, d_ovps.occTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_15, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_25, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &d_ovps.t_val12[iocc1], 1, d_ovps.occTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ1, mc_pair_num, &beta, d_ovps.os_16, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, iocc2 - iocc1, &alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, &beta, d_ovps.os_26, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &d_ovps.t_val12[ivir1], 1, d_ovps.virTau1, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_15, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_25, mc_pair_num), __FILE__, __LINE__);

  cublasStatusAssert(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &d_ovps.t_val12[ivir1], 1, d_ovps.virTau2, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir1, mc_pair_num, &beta, d_ovps.vs_16, mc_pair_num), __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, mc_pair_num, mc_pair_num, ivir2 - ivir1, &alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, &beta, d_ovps.vs_26, mc_pair_num), __FILE__, __LINE__);
  cudaError_t_Assert(cudaThreadSynchronize(), __FILE__, __LINE__);
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

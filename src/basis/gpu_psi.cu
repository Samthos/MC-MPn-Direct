#include <algorithm>
#include <cmath>
#include <iostream>

#include <cuda.h>
#include "cublas_v2.h"

#include "../cublasStatus_t_getErrorString.h"
#include "qc_basis.h"

void Basis::gpu_alloc(int mc_pair_num, Molec &molec) {
  cudaError_t_Assert(cudaMalloc((void **)&d_basis.ao_amplitudes, sizeof(double) * nw_nbf * mc_pair_num * 2), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMalloc((void **)&d_basis.contraction_exp, sizeof(double) * qc_nprm), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void **)&d_basis.contraction_coef, sizeof(double) * qc_nprm), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void **)&d_basis.nw_co, sizeof(double) * nw_nbf * nw_nmo[0]), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMalloc((void **)&d_basis.meta_data, sizeof(BasisMetaData) * qc_nshl), __FILE__, __LINE__);

  cudaError_t_Assert(cudaMemcpy(d_basis.contraction_exp, h_basis.contraction_exp, sizeof(double) * qc_nprm, cudaMemcpyHostToDevice), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(d_basis.contraction_coef, h_basis.contraction_coef, sizeof(double) * qc_nprm, cudaMemcpyHostToDevice), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(d_basis.nw_co, h_basis.nw_co, sizeof(double) * nw_nbf * nw_nmo[0], cudaMemcpyHostToDevice), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(d_basis.meta_data, h_basis.meta_data, sizeof(BasisMetaData) * qc_nshl, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}
void Basis::gpu_free() {
  cudaError_t_Assert(cudaFree(d_basis.ao_amplitudes), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_basis.contraction_exp), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_basis.contraction_coef), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_basis.nw_co), __FILE__, __LINE__);
  cudaError_t_Assert(cudaFree(d_basis.meta_data), __FILE__, __LINE__);
}

__global__ void device_cgs_get(int mc_pair_num, int qc_nshl, BasisData d_basis) {
  int i;
  int iat, iam, ic;
  double dr[3], r2, rad, ang[15];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;  //gives position to use
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;  //gives shell

  if (tidy < qc_nshl && tidx < mc_pair_num) {  //if tidy is less than number of shells
    iat = d_basis.at[tidy];
    iam = d_basis.am[tidy];
    ic = d_basis.isgs[tidy];

    //get position of walker
    dr[0] = d_basis.pos[tidx * 3 + 0];
    dr[1] = d_basis.pos[tidx * 3 + 1];
    dr[2] = d_basis.pos[tidx * 3 + 2];

    //calc diff between walker and atom
    dr[0] = dr[0] - d_basis.apos[iat * 3 + 0];  //x
    dr[1] = dr[1] - d_basis.apos[iat * 3 + 1];  //y
    dr[2] = dr[2] - d_basis.apos[iat * 3 + 2];  //z
    r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

    //calculate amplitude of contracted gaussians
    rad = 0.0;
    for (i = d_basis.stop_list[tidy]; i < d_basis.stop_list[tidy + 1]; i++) {
      rad = rad + exp(-d_basis.contraction_exp[i] * r2) * d_basis.contraction_coef[i];
    }

    if (iam == 0) {
      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad;
    } else if (iam == -1) {
      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad;
      d_basis.ao_amplitudes[(ic + 1) * mc_pair_num + tidx] = rad * dr[0];
      d_basis.ao_amplitudes[(ic + 2) * mc_pair_num + tidx] = rad * dr[1];
      d_basis.ao_amplitudes[(ic + 3) * mc_pair_num + tidx] = rad * dr[2];
    } else if (iam == 1) {
      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad * dr[0];
      d_basis.ao_amplitudes[(ic + 1) * mc_pair_num + tidx] = rad * dr[1];
      d_basis.ao_amplitudes[(ic + 2) * mc_pair_num + tidx] = rad * dr[2];
    } else if (iam == 2) {
      ang[0] = 1.732050807568877 * dr[0] * dr[1];                            // dxy
      ang[1] = 1.732050807568877 * dr[1] * dr[2];                            // dyz
      ang[2] = 0.5 * (2.0 * dr[2] * dr[2] - dr[0] * dr[0] - dr[1] * dr[1]);  // dxx, dyy, dzz
      ang[3] = -1.732050807568877 * dr[0] * dr[2];                           // dxz
      ang[4] = 0.86602540378443 * (dr[0] * dr[0] - dr[1] * dr[1]);           // dxx, dyy

      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad * ang[0];
      d_basis.ao_amplitudes[(ic + 1) * mc_pair_num + tidx] = rad * ang[1];
      d_basis.ao_amplitudes[(ic + 2) * mc_pair_num + tidx] = rad * ang[2];
      d_basis.ao_amplitudes[(ic + 3) * mc_pair_num + tidx] = rad * ang[3];
      d_basis.ao_amplitudes[(ic + 4) * mc_pair_num + tidx] = rad * ang[4];
    } else if (iam == 3) {
      ang[0] = dr[1] * (2.371708245126 * dr[0] * dr[0] - 0.790569415042 * dr[1] * dr[1]);  // xxy, yyy
      ang[1] = 3.872983346207 * dr[0] * dr[1] * dr[2];
      ang[2] = dr[1] * (2.449489742783 * dr[2] * dr[2] - 0.612372435696 * (dr[0] * dr[0] + dr[1] * dr[1]));
      ang[3] = dr[2] * (dr[2] * dr[2] - 1.500000000000 * (dr[0] * dr[0] + dr[1] * dr[1]));
      ang[4] = -dr[0] * (2.449489742783 * dr[2] * dr[2] - 0.612372435696 * (dr[0] * dr[0] + dr[1] * dr[1]));
      ang[5] = dr[2] * 1.936491673104 * (dr[0] * dr[0] - dr[1] * dr[1]);
      ang[6] = dr[0] * (2.371708245126 * dr[1] * dr[1] - 0.790569415042 * dr[0] * dr[0]);

      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad * ang[0];
      d_basis.ao_amplitudes[(ic + 1) * mc_pair_num + tidx] = rad * ang[1];
      d_basis.ao_amplitudes[(ic + 2) * mc_pair_num + tidx] = rad * ang[2];
      d_basis.ao_amplitudes[(ic + 3) * mc_pair_num + tidx] = rad * ang[3];
      d_basis.ao_amplitudes[(ic + 4) * mc_pair_num + tidx] = rad * ang[4];
      d_basis.ao_amplitudes[(ic + 5) * mc_pair_num + tidx] = rad * ang[5];
      d_basis.ao_amplitudes[(ic + 6) * mc_pair_num + tidx] = rad * ang[6];
    } else if (iam == 4) {
      ang[0] = 2.9580398915498085 * (dr[0] * dr[0] * dr[0] * dr[1] - dr[0] * dr[1] * dr[1] * dr[1]);       // xxxy, xyyy
      ang[1] = dr[1] * dr[2] * (6.2749501990055672 * dr[0] * dr[0] - 2.0916500663351894 * dr[1] * dr[1]);  // (2,1,1) (0,3,1)
      ang[2] = dr[0] * dr[1] * 1.1180339887498949 * (-dr[0] * dr[0] - dr[1] * dr[1]) + 6.7082039324993694 * dr[0] * dr[1] * dr[2] * dr[2];
      ang[3] = -2.3717082451262845 * dr[0] * dr[0] * dr[1] * dr[2] - 2.3717082451262845 * dr[1] * dr[1] * dr[1] * dr[2] + 3.1622776601683795 * dr[1] * dr[2] * dr[2] * dr[2];
      ang[4] = 0.375 * (dr[0] * dr[0] * dr[0] * dr[0] + dr[1] * dr[1] * dr[1] * dr[1] + 2.0 * dr[0] * dr[0] * dr[1] * dr[1]) + dr[2] * dr[2] * dr[2] * dr[2] - 3.0 * dr[2] * dr[2] * (dr[0] * dr[0] + dr[1] * dr[1]);
      ang[5] = 2.3717082451262845 * dr[0] * dr[0] * dr[0] * dr[2] + 2.3717082451262845 * dr[0] * dr[1] * dr[1] * dr[2] - 3.1622776601683795 * dr[0] * dr[2] * dr[2] * dr[2];
      ang[6] = 0.55901699437494745 * (dr[1] * dr[1] * dr[1] * dr[1] - dr[0] * dr[0] * dr[0] * dr[0]) + 3.3541019662496847 * dr[2] * dr[2] * (dr[0] * dr[0] - dr[1] * dr[1]);
      ang[7] = dr[0] * dr[2] * (6.2749501990055672 * dr[1] * dr[1] - 2.0916500663351894 * dr[0] * dr[0]);  // (1,2,1) (3,0,1)
      ang[8] = 0.73950997288745213 * (dr[0] * dr[0] * dr[0] * dr[0] + dr[1] * dr[1] * dr[1] * dr[1]) - 4.4370598373247132 * dr[0] * dr[0] * dr[1] * dr[1];

      d_basis.ao_amplitudes[(ic + 0) * mc_pair_num + tidx] = rad * ang[0];
      d_basis.ao_amplitudes[(ic + 1) * mc_pair_num + tidx] = rad * ang[1];
      d_basis.ao_amplitudes[(ic + 2) * mc_pair_num + tidx] = rad * ang[2];
      d_basis.ao_amplitudes[(ic + 3) * mc_pair_num + tidx] = rad * ang[3];
      d_basis.ao_amplitudes[(ic + 4) * mc_pair_num + tidx] = rad * ang[4];
      d_basis.ao_amplitudes[(ic + 5) * mc_pair_num + tidx] = rad * ang[5];
      d_basis.ao_amplitudes[(ic + 6) * mc_pair_num + tidx] = rad * ang[6];
      d_basis.ao_amplitudes[(ic + 7) * mc_pair_num + tidx] = rad * ang[7];
      d_basis.ao_amplitudes[(ic + 8) * mc_pair_num + tidx] = rad * ang[8];
    }
  }
}

void Basis::device_psi_get(double *occ1,
                           double *occ2,
                           double *vir1,
                           double *vir2,
                           double *psi1,
                           double *psi2,
                           double *pos,
                           int mc_pair_num) {
  double alpha = 1.00;
  double beta = 0.00;
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((mc_pair_num * 2 + 255) / 256, qc_nshl, 1);

  //copy pos onto GPU
  cudaError_t_Assert(cudaMemcpy(d_basis.pos, pos, sizeof(double) * mc_pair_num * 6, cudaMemcpyHostToDevice), __FILE__, __LINE__);

  //calculate ao amplitudes
  device_cgs_get<<<gridSize, blockSize>>>(mc_pair_num * 2, qc_nshl, d_basis);
  cudaError_t_Assert(cudaPeekAtLastError(), __FILE__, __LINE__);

  //calculate mo amplitudes
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, iocc2 - iocc1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[0], mc_pair_num * 2,
                                 &d_basis.nw_co[iocc1 * nw_nbf], nw_nbf,
                                 &beta, occ1, mc_pair_num),
                     __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, iocc2 - iocc1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[mc_pair_num], mc_pair_num * 2,
                                 &d_basis.nw_co[iocc1 * nw_nbf], nw_nbf,
                                 &beta, occ2, mc_pair_num),
                     __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, ivir2 - ivir1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[0], mc_pair_num * 2,
                                 &d_basis.nw_co[ivir1 * nw_nbf], nw_nbf,
                                 &beta, vir1, mc_pair_num),
                     __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, ivir2 - ivir1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[mc_pair_num], mc_pair_num * 2,
                                 &d_basis.nw_co[ivir1 * nw_nbf], nw_nbf,
                                 &beta, vir2, mc_pair_num),
                     __FILE__, __LINE__);

  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, ivir2 - iocc1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[0], mc_pair_num * 2,
                                 &d_basis.nw_co[iocc1 * nw_nbf], nw_nbf,
                                 &beta, psi1, mc_pair_num),
                     __FILE__, __LINE__);
  cublasStatusAssert(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 mc_pair_num, ivir2 - iocc1, nw_nbf, &alpha,
                                 &d_basis.ao_amplitudes[mc_pair_num], mc_pair_num * 2,
                                 &d_basis.nw_co[iocc1 * nw_nbf], nw_nbf,
                                 &beta, psi2, mc_pair_num),
                     __FILE__, __LINE__);

  cudaError_t_Assert(cudaThreadSynchronize(), __FILE__, __LINE__);
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

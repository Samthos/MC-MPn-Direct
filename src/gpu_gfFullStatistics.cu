#include <iostream>
#include <vector>

#include "cublas_v2.h"

#include "cublasStatus_t_getErrorString.h"

#include "qc_monte.h"

__global__ void squareElements(int m, int n, double* A) {
  // m is number of rows of A
  // n is number of columsn of A
  // A is an m*n column major matrix
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;  // row index
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;  // column index
  if (tidx < n && tidy < m) {
    int index = tidy * m + tidx;
    A[index] *= A[index];
  }
}

void QC_monte::mc_gf_statistics(int step,
                                std::vector<std::vector<double>>& qep,
                                std::vector<std::vector<double*>>& en,
                                std::vector<std::vector<std::vector<double*>>>& enBlock,
                                std::vector<std::vector<std::vector<double*>>>& enEx1,
                                std::vector<std::vector<std::vector<double*>>>& enEx2) {
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);
  double alpha, beta;
  int blockPower2, blockStep;
  int offset;

  // initialize block and grid size variables
  dim3 blockSize(8, 8, 1);
  dim3 gridSize((ivir2 - iocc1 + 7) / 8, (ivir2 - iocc1 + 7) / 8, 1);

  for (auto band = 0; band < iops.iopns[KEYS::NUM_BAND]; band++) {
    offset = (ivir2 - iocc1) * (iocc2 - iocc1 - offBand + band) + (iocc2 - iocc1 - offBand + band);
    for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
      cudaError_t_Assert(cudaMemcpy(&qep[band][diff], en[band][diff] + offset, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

      blockPower2 = 1;
      for (auto block = 0; block < iops.iopns[KEYS::NBLOCK]; block++) {
        blockStep = (step - 1) % blockPower2 + 1;

        // enBlock[i][j] = en / step + (step-1)*enBlock[i][j]/(step) i.e update first moment
        alpha = 1.0 / static_cast<double>(blockStep);
        beta = (static_cast<double>(blockStep) - 1.0) / static_cast<double>(blockStep);
        cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                       ivir2 - iocc1, ivir2 - iocc1,
                                       &alpha, en[band][diff], ivir2 - iocc1,
                                       &beta, enBlock[band][diff][block], ivir2 - iocc1,
                                       enBlock[band][diff][block], ivir2 - iocc1),
                           __FILE__, __LINE__);

        if ((step & (blockPower2 - 1)) == 0) {  //if block is filled -> accumulate
          blockStep = step / blockPower2;

          // enEx1[i] = en / step + (step-1)*enEx1[i]/(step) i.e update first momenT
          alpha = 1.0 / static_cast<double>(blockStep);
          beta = (static_cast<double>(blockStep) - 1.0) / static_cast<double>(blockStep);
          cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         ivir2 - iocc1, ivir2 - iocc1,
                                         &alpha, enBlock[band][diff][block], ivir2 - iocc1,
                                         &beta, enEx1[band][diff][block], ivir2 - iocc1,
                                         enEx1[band][diff][block], ivir2 - iocc1),
                             __FILE__, __LINE__);

          // en[i][j] = en[i][j]^2
          squareElements<<<gridSize, blockSize>>>(ivir2 - iocc1, ivir2 - iocc1, enBlock[band][diff][block]);
          cudaError_t_Assert(cudaPeekAtLastError(), __FILE__, __LINE__);

          // enEx1 = en/step + (step-1)*enEx2/(step) i.e update second momenT
          cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         ivir2 - iocc1, ivir2 - iocc1,
                                         &alpha, enBlock[band][diff][block], ivir2 - iocc1,
                                         &beta, enEx2[band][diff][block], ivir2 - iocc1,
                                         enEx2[band][diff][block], ivir2 - iocc1),
                             __FILE__, __LINE__);

          // zero block;
          cudaError_t_Assert(cudaMemset(enBlock[band][diff][block], 0, sizeof(double) * (ivir2 - iocc1) * (ivir2 - iocc1)), __FILE__, __LINE__);
        }
        blockPower2 *= 2;
      }
    }
  }
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

void QC_monte::mc_gf2_statistics(int band, int step) {
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);
  double diffMultiplier = 1.0;
  double alpha, beta;

  // initialize block and grid size variables
  dim3 blockSize(8, 8, 1);
  dim3 gridSize((ivir2 - iocc1 + 7) / 8, (ivir2 - iocc1 + 7) / 8, 1);

  for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
    // en2 = diffmultiplier * en2pCore + en2
    alpha = diffMultiplier;
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en2p, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en2[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en2[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en2 = (-1)^i diffMultiplier * en2mCore + en2
    if (diff % 2 == 1) {
      alpha = -diffMultiplier;
    } else {
      alpha = diffMultiplier;
    }
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en2m, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en2[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en2[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    diffMultiplier *= ovps.xx1;
  }
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

void QC_monte::mc_gf3_statistics(int band, int step) {
  cublasHandle_t handle;
  cublasStatusAssert(cublasCreate(&handle), __FILE__, __LINE__);
  std::array<double, 3> diffMultiplier;
  double alpha;
  double beta;

  std::fill(diffMultiplier.begin(), diffMultiplier.end(), 1.0);

  // initialize block and grid size variables
  dim3 blockSize(8, 8, 1);
  dim3 gridSize((ivir2 - iocc1 + 7) / 8, (ivir2 - iocc1 + 7) / 8, 1);

  for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
    // en3 = diffMultipllier[0] en3_1mCore
    alpha = diffMultiplier[0];
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_1p, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en3 = diffMultipllier[0] en3_2mCore + en3
    alpha = diffMultiplier[1];
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_2p, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en3 = diffMultipllier[0] en3_12mCore + en3
    alpha = diffMultiplier[2];
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_12p, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en3 = en3_1mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[0];
    } else {
      alpha = diffMultiplier[0];
    }
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_1m, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en3 = en3_2mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[1];
    } else {
      alpha = diffMultiplier[1];
    }
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_2m, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    // en3 = en3_12mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[2];
    } else {
      alpha = diffMultiplier[2];
    }
    beta = 1.0;
    cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   ivir2 - iocc1, ivir2 - iocc1,
                                   &alpha, ovps.d_ovps.en3_12m, ivir2 - iocc1,
                                   &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                   ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                       __FILE__, __LINE__);

    if (diff == 0) {
      // en3 = en3_c + en3
      alpha = 1.0;
      beta = 1.0;
      cublasStatusAssert(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     ivir2 - iocc1, ivir2 - iocc1,
                                     &alpha, ovps.d_ovps.en3_c, ivir2 - iocc1,
                                     &beta, ovps.d_ovps.en3[band][diff], ivir2 - iocc1,
                                     ovps.d_ovps.en3[band][diff], ivir2 - iocc1),
                         __FILE__, __LINE__);
    }
    diffMultiplier[0] *= ovps.xx1;
    diffMultiplier[1] *= ovps.xx2;
    diffMultiplier[2] *= ovps.xx1 * ovps.xx2;
  }
  cublasStatusAssert(cublasDestroy(handle), __FILE__, __LINE__);
}

void QC_monte::mc_gf_copy(std::vector<double>& ex1, std::vector<double>& ex2, double* d_ex1, double* d_ex2) {
  cudaError_t_Assert(cudaMemcpy(ex1.data(), d_ex1, sizeof(double) * ex1.size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
  cudaError_t_Assert(cudaMemcpy(ex2.data(), d_ex2, sizeof(double) * ex2.size(), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

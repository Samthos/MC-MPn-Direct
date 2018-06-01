#include <iostream>
#include <string>
#include <cuda.h>
#include "cublas_v2.h"

#include "cublasStatus_t_getErrorString.h"

std::string cublasStatus_t_getErrorString(const cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return "The operation completed successfully.";
  } else if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "The cuBLAS library was not initialized.";
  } else if (status == CUBLAS_STATUS_ALLOC_FAILED) {
    return "Resource allocation failed inside the cuBLAS library.";
  } else if (status == CUBLAS_STATUS_INVALID_VALUE) {
    return "An unsupported value or parameter was passed to the function.";
  } else if (status == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "The function requires a feature absent from the device architecture.";
  } else if (status == CUBLAS_STATUS_MAPPING_ERROR) {
    return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";
  } else if (status == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "The GPU program failed to execute.";
  } else if (status == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "An internal cuBLAS operation failed.";
  } else if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "The functionnality requested is not supported";
  } else if (status == CUBLAS_STATUS_LICENSE_ERROR) {
    return "The functionnality requested requires some license";
  } else {
    return "Error code not recognized";
  }
}

void cublasStatusAssert(const cublasStatus_t status, std::string file, int line, bool abort) {
#ifndef NDEBUG
  std::cerr << "From file " << file << " on line " << line << ": " << cublasStatus_t_getErrorString(status) << std::endl;
#endif  // NDEBUG

  if (status != CUBLAS_STATUS_SUCCESS) {
#ifdef NDEBUG
    std::cerr << "From file " << file << " on line " << line << ": " << cublasStatus_t_getErrorString(status) << std::endl;
#endif  // NDEBUG
    if (abort) {
      exit(status);
    }
  }
}

void cudaError_t_Assert(cudaError_t status, std::string file, int line, bool abort) {
#ifndef NDEBUG
  std::cerr << "From file " << file << " on line " << line << ": " << cudaGetErrorString(status) << std::endl;
#endif  // NDEBUG

  if (status != cudaSuccess) {
#ifdef NDEBUG
    std::cerr << "From file " << file << " on line " << line << ": " << cudaGetErrorString(status) << std::endl;
#endif  // NDEBUG
    if (abort) {
      exit(status);
    }
  }
}

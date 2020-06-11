#include <string>

#include <cuda.h>
#include "cublas_v2.h"

#ifndef CUBLASSTATUS_T_GETERRORSTRING_H_
#define CUBLASSTATUS_T_GETERRORSTRING_H_
std::string cublasStatus_t_getErrorString(const cublasStatus_t);
void cublasStatusAssert(const cublasStatus_t status, std::string file, int line, bool abort = true);
void cudaError_t_Assert(const cudaError_t status, std::string file, int line, bool abort = true);
#endif  // CUBLASSTATUS_T_GETERRORSTRING_H_

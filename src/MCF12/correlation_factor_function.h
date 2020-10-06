#ifndef CORRELATION_FACTOR_FUNCTION_H_
#define CORRELATION_FACTOR_FUNCTION_H_

#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#define SOURCE_FILE "correlation_factor_function.imp.h"
#include "correlation_factor_patterns.h"
#undef SOURCE_FILE

#undef HOSTDEVICE
#endif // CORRELATION_FACTOR_FUNCTION_H_

#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include "cublas_v2.h"
#include "device_mp2_functional.h"

template <int CVMP2>
Device_MP2_Functional<CVMP2>::Device_MP2_Functional(int electron_pairs) : 
  Standard_MP_Functional<thrust::device_vector, thrust::device_allocator>(CVMP2 * (CVMP2+1), 1, "22"),
  vector_size(electron_pairs),
  matrix_size(vector_size * vector_size),
  ctrl(n_control_variates),
  o_direct(matrix_size),
  o_exchange(matrix_size),
  v_direct(matrix_size),
  v_exchange(matrix_size),
  scratch_matrix(matrix_size),
  scratch_vector(2 * vector_size),
  d_en_ctrl(8),
  h_en_ctrl(8)
{ 
  block_size = dim3(16, 16, 1);
  grid_size = dim3(
      (vector_size + block_size.x - 1) / block_size.x, 
      (vector_size + block_size.y - 1) / block_size.y, 
      1);
  cublasCreate(&handle);
}

template <int CVMP2>
Device_MP2_Functional<CVMP2>::~Device_MP2_Functional() {
  cublasDestroy(handle);
}

__global__ void m_m_add_mul(double alpha, double* A, double *B, double* C, int size) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid  = tidy * size + tidx;
  if(tidx < size && tidy < tidx) {
    C[tid] = alpha * A[tid] * B[tid] + C[tid];
  }
}

__global__ 
void mp2_functional_kernal(
    const double *o_direct,
    const double *o_exchange,
    const double *v_term,
    double *scratch_matrix,
    int size) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid  = tidy * size + tidx;
  if(tidx < size && tidy < tidx) {
    scratch_matrix[tid] -= 2.0 * o_direct[tid] * v_term[tid];
    scratch_matrix[tid] += o_exchange[tid] * v_term[tid];
  }
}

template <int CVMP2>
void Device_MP2_Functional<CVMP2>::prep_arrays(OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list) {
  en2 = 0.0;
  std::fill(ctrl.begin(), ctrl.end(), 0.0);
  thrust::transform(ovps.o_set[0][0].s_11.begin(), ovps.o_set[0][0].s_11.end(), ovps.o_set[0][0].s_22.begin(), o_direct.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.o_set[0][0].s_12.begin(), ovps.o_set[0][0].s_12.end(), ovps.o_set[0][0].s_21.begin(), o_exchange.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.v_set[0][0].s_11.begin(), ovps.v_set[0][0].s_11.end(), ovps.v_set[0][0].s_22.begin(), v_direct.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.v_set[0][0].s_12.begin(), ovps.v_set[0][0].s_12.end(), ovps.v_set[0][0].s_21.begin(), v_exchange.begin(), thrust::multiplies<double>());
}

template <int CVMP2>
void Device_MP2_Functional<CVMP2>::cv_energy_helper(int offset, const vector_double& rv_inverse_weight) {
  this->blas_wrapper.dgemm(false, false,
      vector_size, 2, vector_size,
      1.0,
      scratch_matrix, vector_size,
      rv_inverse_weight, vector_size,
      0.0,
      scratch_vector, vector_size);

  this->blas_wrapper.dgemm(true, false,
      2, 2, vector_size,
      1.0,
      scratch_vector, 0, vector_size,
      rv_inverse_weight, 0, vector_size,
      0.0,
      d_en_ctrl, offset * 2, 4);
}

template <int CVMP2>
void Device_MP2_Functional<CVMP2>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau) {
  prep_arrays(ovps, electron_pair_list);

  thrust::fill(scratch_matrix.begin(), scratch_matrix.end(), 0.0);
  m_m_add_mul<<<grid_size, block_size>>>(1.0, o_direct.data().get(), v_direct.data().get(), scratch_matrix.data().get(), vector_size);
  m_m_add_mul<<<grid_size, block_size>>>(1.0, o_exchange.data().get(), v_exchange.data().get(), scratch_matrix.data().get(), vector_size);
  cv_energy_helper(0, electron_pair_list->rv_inverse_weight);

  thrust::fill(scratch_matrix.begin(), scratch_matrix.end(), 0.0);
  m_m_add_mul<<<grid_size, block_size>>>(1.0, o_direct.data().get(),   v_exchange.data().get(), scratch_matrix.data().get(), vector_size);
  m_m_add_mul<<<grid_size, block_size>>>(1.0, o_exchange.data().get(), v_direct.data().get(), scratch_matrix.data().get(), vector_size);
  cv_energy_helper(1, electron_pair_list->rv_inverse_weight);

  thrust::copy(d_en_ctrl.begin(), d_en_ctrl.end(), h_en_ctrl.begin());
  en2 = h_en_ctrl[2] - 2.0 * h_en_ctrl[0];
  ctrl[0] = h_en_ctrl[1]; 
  ctrl[1] = h_en_ctrl[3];
  if (CVMP2 >= 2) {
    ctrl[2] = h_en_ctrl[4];
    ctrl[3] = h_en_ctrl[6];
    ctrl[4] = h_en_ctrl[5];
    ctrl[5] = h_en_ctrl[7];
  }

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp = emp + en2 * tau_wgt;
  if (CVMP2 >= 1) {
    thrust::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * tau_wgt; });
  }
}

template <>
void Device_MP2_Functional<0>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau) {
  prep_arrays(ovps, electron_pair_list);
  thrust::fill(scratch_matrix.begin(), scratch_matrix.end(), 0.0);
  mp2_functional_kernal<<<grid_size, block_size>>>(o_direct.data().get(), o_exchange.data().get(), v_direct.data().get(), scratch_matrix.data().get(), vector_size);
  mp2_functional_kernal<<<grid_size, block_size>>>(o_exchange.data().get(), o_direct.data().get(), v_exchange.data().get(), scratch_matrix.data().get(), vector_size);
  cv_energy_helper(0, electron_pair_list->rv_inverse_weight);
  thrust::copy(d_en_ctrl.begin(), d_en_ctrl.end(), h_en_ctrl.begin());
  en2 = h_en_ctrl[0];

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp = emp + en2 * tau_wgt;
}

#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include "device_mp2_functional.h"

template <int CVMP2>
Device_MP2_Functional<CVMP2>::Device_MP2_Functional(int electron_pairs) : 
  Standard_MP_Functional<thrust::device_vector, thrust::device_allocator>(CVMP2 * (CVMP2+1), 1, "22"),
  vector_size(electron_pairs),
  matrix_size(vector_size * vector_size),
  grid_size((matrix_size + block_size - 1) / block_size),
  o_direct(matrix_size),
  o_exchange(matrix_size),
  v_term(matrix_size),
  scratch_matrix(matrix_size),
  scratch_vector(vector_size),
  inverse_weight(vector_size),
  rv(vector_size)
{ }

__global__ 
void mp2_functional_kernal(
    const double *o_direct,
    const double *o_exchange,
    const double *v_term,
    double *scratch_matrix,
    int size) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < size) {
    scratch_matrix[tidx] -= 2.0 * o_direct[tidx] * v_term[tidx];
    scratch_matrix[tidx] += o_exchange[tidx] * v_term[tidx];
  }
}

template <int CVMP2>
void Device_MP2_Functional<CVMP2>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List* electron_pair_list, Tau* tau) {
  double en2 = 0.0;
  std::vector<double> ctrl(control.size(), 0.0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  int vector_size = electron_pair_list->size();
  int matrix_size = vector_size * vector_size;

  int block_size = 16;
  int grid_size = (matrix_size + block_size - 1) / block_size;

  thrust::device_vector<double> o_direct(matrix_size);
  thrust::device_vector<double> o_exchange(matrix_size);
  thrust::device_vector<double> v_term(matrix_size);
  thrust::device_vector<double> scratch_matrix(matrix_size, 0.0);
  thrust::device_vector<double> scratch_vector(vector_size, 0.0);
  thrust::device_vector<double> inverse_weight(electron_pair_list->inverse_weight);
  thrust::device_vector<double> rv(electron_pair_list->rv);

  // uncomment when scratch variables moved to class members
  // thrust::fill(scratch_matrix.begin(), scratch_matrix.end(), 0.0);

  thrust::transform(ovps.o_set[0][0].s_11.begin(), ovps.o_set[0][0].s_11.end(), ovps.o_set[0][0].s_22.begin(), o_direct.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.o_set[0][0].s_12.begin(), ovps.o_set[0][0].s_12.end(), ovps.o_set[0][0].s_21.begin(), o_exchange.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.v_set[0][0].s_11.begin(), ovps.v_set[0][0].s_11.end(), ovps.v_set[0][0].s_22.begin(), v_term.begin(), thrust::multiplies<double>());
  mp2_functional_kernal<<<grid_size, block_size>>>(o_direct.data().get(), o_exchange.data().get(), v_term.data().get(), scratch_matrix.data().get(), matrix_size);
  thrust::transform(ovps.v_set[0][0].s_12.begin(), ovps.v_set[0][0].s_12.end(), ovps.v_set[0][0].s_21.begin(), v_term.begin(), thrust::multiplies<double>());
  mp2_functional_kernal<<<grid_size, block_size>>>(o_exchange.data().get(), o_direct.data().get(), v_term.data().get(), scratch_matrix.data().get(), matrix_size);

  double alpha = 1.0;
  double beta  = 0.0;
  cublasDgemv(handle, CUBLAS_OP_N,
      vector_size, vector_size,
      &alpha,
      scratch_matrix.data().get(), vector_size,
      rv.data().get(), 1,
      &beta,
      scratch_vector.data().get(), 1);
  cublasDdot(handle, vector_size,
      scratch_vector.data().get(), 1,
      rv.data().get(), 1, 
      &en2);

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  tau_wgt /= 2;
  emp = emp + en2 * tau_wgt;

  cublasDestroy(handle);
// for (auto it = 0; it != electron_pair_list->size(); it++) {
//   en_rj.fill(0.0);
//   if (CVMP2 >= 2) {
//     en_wj.fill(0.0);
//   }
//
//   for (auto jt = it + 1; jt !=  electron_pair_list->size(); jt++) {
//     auto ijIndex = it * electron_pair_list->size() + jt;
//     en[0] = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
//     en[1] = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
//
//     en[0] = en[0] + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
//     en[1] = en[1] + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
//
//     std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt];});
//     if (CVMP2 >= 2) {
//       std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt];});
//     }
//   }
//   en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
//   if (CVMP2 >= 1) {
//     std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
//   }
//   if (CVMP2 >= 2) {
//     std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
//     std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
//   }
// }
// auto tau_wgt = tau->get_wgt(1);
// tau_wgt /= static_cast<double>(electron_pair_list->size());
// tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
// emp = emp + en2 * tau_wgt;
// if (CVMP2 >= 1) {
//   std::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * tau_wgt; });
// }
}

template <>
void Device_MP2_Functional<0>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List* electron_pair_list, Tau* tau) {
  double en2 = 0.0;
  std::vector<double> ctrl(control.size(), 0.0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // uncomment when scratch variables moved to class members
  thrust::fill(scratch_matrix.begin(), scratch_matrix.end(), 0.0);
  thrust::copy(electron_pair_list->rv.begin(), electron_pair_list->rv.end(), rv.begin());
  thrust::copy(electron_pair_list->inverse_weight.begin(), electron_pair_list->inverse_weight.end(), inverse_weight.begin());

  thrust::transform(ovps.o_set[0][0].s_11.begin(), ovps.o_set[0][0].s_11.end(), ovps.o_set[0][0].s_22.begin(), o_direct.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.o_set[0][0].s_12.begin(), ovps.o_set[0][0].s_12.end(), ovps.o_set[0][0].s_21.begin(), o_exchange.begin(), thrust::multiplies<double>());
  thrust::transform(ovps.v_set[0][0].s_11.begin(), ovps.v_set[0][0].s_11.end(), ovps.v_set[0][0].s_22.begin(), v_term.begin(), thrust::multiplies<double>());
  mp2_functional_kernal<<<grid_size, block_size>>>(o_direct.data().get(), o_exchange.data().get(), v_term.data().get(), scratch_matrix.data().get(), matrix_size);
  thrust::transform(ovps.v_set[0][0].s_12.begin(), ovps.v_set[0][0].s_12.end(), ovps.v_set[0][0].s_21.begin(), v_term.begin(), thrust::multiplies<double>());
  mp2_functional_kernal<<<grid_size, block_size>>>(o_exchange.data().get(), o_direct.data().get(), v_term.data().get(), scratch_matrix.data().get(), matrix_size);

  double alpha = 1.0;
  double beta  = 0.0;
  cublasDgemv(handle, CUBLAS_OP_N,
      vector_size, vector_size,
      &alpha,
      scratch_matrix.data().get(), vector_size,
      rv.data().get(), 1,
      &beta,
      scratch_vector.data().get(), 1);
  cublasDdot(handle, vector_size,
      scratch_vector.data().get(), 1,
      rv.data().get(), 1, 
      &en2);

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  tau_wgt /= 2;
  emp = emp + en2 * tau_wgt;

  cublasDestroy(handle);
// for (auto it = 0; it != electron_pair_list->size(); it++) {
//   en_rj.fill(0.0);
//   if (CVMP2 >= 2) {
//     en_wj.fill(0.0);
//   }
//
//   for (auto jt = it + 1; jt !=  electron_pair_list->size(); jt++) {
//     auto ijIndex = it * electron_pair_list->size() + jt;
//     en[0] = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
//     en[1] = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
//
//     en[0] = en[0] + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
//     en[1] = en[1] + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
//
//     std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt];});
//     if (CVMP2 >= 2) {
//       std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt];});
//     }
//   }
//   en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
//   if (CVMP2 >= 1) {
//     std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
//   }
//   if (CVMP2 >= 2) {
//     std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
//     std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
//   }
// }
// auto tau_wgt = tau->get_wgt(1);
// tau_wgt /= static_cast<double>(electron_pair_list->size());
// tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
// emp = emp + en2 * tau_wgt;
// if (CVMP2 >= 1) {
//   std::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * tau_wgt; });
// }
}

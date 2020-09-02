#ifndef DEVICE_MP2_Functional_H_
#define DEVICE_MP2_Functional_H_

#include <thrust/device_vector.h>
#include "cublas_v2.h"

#include "mp_functional.h"

template <int CVMP2> 
class Device_MP2_Functional : public Standard_MP_Functional<thrust::device_vector, thrust::device_allocator> {
 public:
   Device_MP2_Functional(int);
   ~Device_MP2_Functional();
   void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List*, Tau*) override;

 private:
  void prep_arrays(OVPS_Type&, Electron_Pair_List*);
  double cv_energy_helper(int);

  int vector_size;
  int matrix_size;
  dim3 block_size;
  dim3 grid_size;

  cublasHandle_t handle;

  double en2;
  std::vector<double> ctrl;
  thrust::device_vector<double> o_direct;
  thrust::device_vector<double> o_exchange;
  thrust::device_vector<double> v_direct;
  thrust::device_vector<double> v_exchange;
  thrust::device_vector<double> scratch_matrix;
  thrust::device_vector<double> scratch_vector;
  thrust::device_vector<double> inverse_weight;
  thrust::device_vector<double> rv;
};

template <> void Device_MP2_Functional<0>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List* electron_pair_list, Tau* tau);

template class Device_MP2_Functional<0>;
template class Device_MP2_Functional<1>;
template class Device_MP2_Functional<2>;

#endif  // DEVICE_MP2_Functional_H_

#ifndef DEVICE_MP2_Functional_H_
#define DEVICE_MP2_Functional_H_

#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include "cublas_v2.h"

#include "mp_functional.h"

template <int CVMP2> 
class Device_MP2_Functional : public Standard_MP_Functional<thrust::device_vector, thrust::device_allocator> {
  typedef Electron_Pair_List_Device Electron_Pair_List_Type;
  typedef thrust::device_vector<double, thrust::device_allocator<double>> vector_double;

 public:
   Device_MP2_Functional(int);
   ~Device_MP2_Functional();
   void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List_Type*, Tau*) override;

 private:
  void prep_arrays(OVPS_Type&, Electron_Pair_List_Type*);
  void cv_energy_helper(int, const vector_double&);

  int vector_size;
  int matrix_size;
  dim3 block_size;
  dim3 grid_size;

  cublasHandle_t handle;

  double en2;
  std::vector<double> ctrl;
  vector_double o_direct;
  vector_double o_exchange;
  vector_double v_direct;
  vector_double v_exchange;
  vector_double scratch_matrix;
  vector_double scratch_vector;
  vector_double d_en_ctrl;
  std::vector<double> h_en_ctrl;
};

template <> void Device_MP2_Functional<0>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau);

template class Device_MP2_Functional<0>;
template class Device_MP2_Functional<1>;
template class Device_MP2_Functional<2>;

#endif  // DEVICE_MP2_Functional_H_

#ifndef MP2_F12_V_H_
#define MP2_F12_V_H_

#include <array>
#include <unordered_map>

#include "../qc_input.h"
#include "../basis/basis.h"
#include "electron_pair_list.h"
#include "electron_list.h"
#include "F12_Traces.h"
#include "correlation_factor_data.h"

#include "../MCMP/mp_functional.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class MP2_F12_V : public F12_MP_Functional<Container, Allocator> {
 protected:
  typedef Container<double, Allocator<double>> vector_double;

  typedef Blas_Wrapper<Container, Allocator> Blas_Wrapper_Type;
  typedef Correlation_Factor_Data<Container, Allocator> Correlation_Factor_Data_Type;
  typedef Electron_List<Container, Allocator> Electron_List_Type;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef F12_Traces<Container, Allocator> F12_Traces_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

 public:
  explicit MP2_F12_V(const IOPs& iops, std::string extension="f12_V");
  ~MP2_F12_V();
  virtual void energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) override;

 protected:
  double calculate_v_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_v_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_v_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_v_4e_help(
      vector_double& R_ip_io, vector_double& R_ip_jo, 
      const vector_double& S_io_jo,
      const vector_double& S_ip_io_1, const vector_double& S_ip_io_2,
      const vector_double& S_ip_jo_1, const vector_double& S_ip_jo_2,
      const vector_double& S_ip,
      size_t size, size_t size_ep);
  void calculate_v(double& emp, std::vector<double>& control,
      std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  //define the amplitudes
  static constexpr double a1 = 3.0/8.0;
  static constexpr double a2 = 1.0/8.0;

  // auto-generate the correct coefficients
  static constexpr double c1 = 2.0*a1-a2;
  static constexpr double c2 = 2.0*a2-a1;

  Blas_Wrapper_Type blas_wrapper;
  F12_Traces_Type traces;
  std::shared_ptr<Correlation_Factor_Data_Type> correlation_factor;

  vector_double T_ip_io;
  vector_double T_ip_jo;
  vector_double T_io_jo;
  vector_double T_ip;
  vector_double T_io;

  double nsamp_pair;
  double nsamp_one_1;
  double nsamp_one_2;
};

template class MP2_F12_V<std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class MP2_F12_V<thrust::device_vector, thrust::device_allocator>;
#endif
#endif  // MP2_F12_V_H_

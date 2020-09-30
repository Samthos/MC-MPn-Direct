//
// Created by aedoran on 1/2/20.
//

#ifndef F12_METHODS_SRC_F12_TRACES_H_
#define F12_METHODS_SRC_F12_TRACES_H_

#include <unordered_map>
#include <vector>
#include "blas_wrapper.h"
#include "../basis/wavefunction.h"
#include "electron_pair_list.h"
#include "electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class F12_Traces {
  typedef Blas_Wrapper<Container, Allocator> Blas_Wrapper_Type;
  typedef Container<Point, Allocator<Point>> vector_Point;
  typedef Container<double, Allocator<double>> vector_double;
  typedef Electron_List<Container, Allocator> Electron_List_Type;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

 public:
  F12_Traces(int electron_pairs_, int electrons_);

  // wavefunction shim
  int electron_pairs, electrons;

  // traces of single electrons with themselves
  vector_double op11;

  // traces of single electrons with single electrons
  vector_double op12;
  vector_double ok12;
  vector_double ov12;

  // traces of electrons pairs with themselves
  vector_double p11;
  vector_double p12;
  vector_double p22;
  vector_double k12;

  // traces of electron pairs with single electrons
  vector_double p13;
  vector_double k13;
  vector_double v13;
  vector_double p23;
  vector_double k23;
  vector_double v23;

  // derivative traces of electron pairs with themselves
  vector_double delta_pos;
  vector_double dp11;
  vector_double dp12;
  vector_double dp21;
  vector_double dp22;

  // derivative traces of electron pairs with electrons
  vector_double dp31;
  vector_double dp32;

  // extra one electron traces
  vector_double ds_p11;
  vector_double ds_p12;
  vector_double ds_p21;
  vector_double ds_p22;
  vector_double ds_p31;
  vector_double ds_p32;

  void update_v(std::unordered_map<int, Wavefunction_Type>& wavefunctions);
  void update_bx(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  void update_bx_fd_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_List_Type* electron_list);

 private:
  Blas_Wrapper_Type blas_wrapper;

  void build_one_e_one_e_traces(const Wavefunction_Type& electron_psi);
  void build_two_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2);
  void build_two_e_one_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2, const Wavefunction_Type& electron_psi);

  void build_delta_pos(const vector_Point&, const vector_Point&);
  void build_two_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list);
  void build_two_e_one_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
};

template <> void F12_Traces<std::vector, std::allocator>::build_delta_pos(const vector_Point&, const vector_Point&);
template class F12_Traces<std::vector, std::allocator>;

#ifdef HAVE_CUDA
template <> void F12_Traces<thrust::device_vector, thrust::device_allocator>::build_delta_pos(const vector_Point&, const vector_Point&);
template class F12_Traces<thrust::device_vector, thrust::device_allocator>;
#endif 
#endif //F12_METHODS_SRC_F12_TRACES_H_

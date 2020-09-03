//
// Created by aedoran on 1/2/20.
//

#ifndef F12_METHODS_SRC_F12_TRACES_H_
#define F12_METHODS_SRC_F12_TRACES_H_

#include <unordered_map>
#include <vector>
#include "../basis/wavefunction.h"
#include "electron_pair_list.h"
#include "electron_list.h"

class F12_Traces {
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List Electron_List_Type;
  typedef Wavefunction_Host Wavefunction_Type;
 public:
  F12_Traces(int electron_pairs_, int electrons_);

  // wavefunction shim
  int electron_pairs, electrons;

  // traces of single electrons with themselves
  std::vector<double> op11;

  // traces of single electrons with single electrons
  std::vector<double> op12;
  std::vector<double> ok12;
  std::vector<double> ov12;

  // traces of electrons pairs with themselves
  std::vector<double> p11;
  std::vector<double> p12;
  std::vector<double> p22;
  std::vector<double> k12;
  std::vector<double> dp11;
  std::vector<double> dp12;
  std::vector<double> dp21;
  std::vector<double> dp22;

  // traces of electron pairs with single electrons
  std::vector<double> p13;
  std::vector<double> k13;
  std::vector<double> v13;
  std::vector<double> dp31;
  std::vector<double> p23;
  std::vector<double> k23;
  std::vector<double> v23;
  std::vector<double> dp32;

  // extra one electron traces
  std::vector<std::vector<double>> ds_p11;
  std::vector<std::vector<double>> ds_p12;
  std::vector<std::vector<double>> ds_p21;
  std::vector<std::vector<double>> ds_p22;
  std::vector<std::vector<std::vector<double>>> ds_p31;
  std::vector<std::vector<std::vector<double>>> ds_p32;

  void update_v(std::unordered_map<int, Wavefunction_Type>& wavefunctions);
  void update_bx(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  void update_bx_fd_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_List_Type* electron_list);

 private:
  void build_one_e_one_e_traces(const Wavefunction_Type& electron_psi);
  void build_two_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2);
  void build_two_e_one_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2, const Wavefunction_Type& electron_psi);

  void build_two_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list);
  void build_two_e_one_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
};

#endif //F12_METHODS_SRC_F12_TRACES_H_

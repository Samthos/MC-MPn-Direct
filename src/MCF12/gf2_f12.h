#ifndef GF2F12_H_
#define GF2F12_H_

#include <array>
#include <unordered_map>

#include "../qc_input.h"
#include "../basis/qc_basis.h"
#include "../electron_pair_list.h"
#include "../electron_list.h"

#include "../MCGF/qc_mcgf.h"
#include "F12_Traces.h"
#include "correlation_factors.h"

class GF2_F12_V : public MCGF {
 public:
  explicit GF2_F12_V(IOPs& iops, Basis& basis);
  ~GF2_F12_V();
  void energy_f12(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction>&,
     Electron_Pair_List*, Electron_List*);


 protected:
  void core(OVPs& ovps, Electron_Pair_List* electron_pair_list);
  void energy_no_diff(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction>&,
     Electron_Pair_List*, Tau*);
  void energy_diff(std::vector<std::vector<double>>&,
     std::unordered_map<int, Wavefunction>&,
     Electron_Pair_List*, Tau*);
  //define the amplitudes
  static constexpr double a1 = 3.0/8.0;
  static constexpr double a2 = 1.0/8.0;

  // auto-generate the correct coefficients
  static constexpr double c1 = a1-0.5*a2;
  static constexpr double c2 = a2-0.5*a1;

  F12_Traces traces;
  Correlation_Factor* correlation_factor;
  double nsamp_pair;
  double nsamp_one_1;
  double nsamp_one_2;
};

/*
class GF2F12_VBX_Engine : public GF2F12_V_Engine {
 public:
  explicit GF2F12_VBX_Engine(const IOPs& iops, const Basis& basis) : GF2F12_V_Engine(iops, basis) {
    nsamp_one_3 = nsamp_one_2 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-2);
    nsamp_one_4 = nsamp_one_3 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-3);
  }
  double calculate_vbx(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

 protected:
  double calculate_bx(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
  void zero();
  void calculate_bx_t_fa(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fb(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fc(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fd(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_k   (const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void normalize();

  static constexpr double c3 = 2.0*(a1*a1+a2*a2-a1*a2);
  static constexpr double c4 = 4*a1*a2-a1*a1-a2*a2;

  double nsamp_one_3;
  double nsamp_one_4;

  std::array<double, 2> direct_1_pair_0_one_ints;
  std::array<double, 2> direct_0_pair_2_one_ints;
  std::array<double, 3> direct_1_pair_1_one_ints;
  std::array<double, 2> direct_0_pair_3_one_ints;
  std::array<double, 6> direct_1_pair_2_one_ints;
  std::array<double, 4> direct_0_pair_4_one_ints;
  std::array<double, 1> direct_1_pair_3_one_ints;

  std::array<double, 2> xchang_1_pair_0_one_ints;
  std::array<double, 2> xchang_0_pair_2_one_ints;
  std::array<double, 3> xchang_1_pair_1_one_ints;
  std::array<double, 2> xchang_0_pair_3_one_ints;
  std::array<double, 6> xchang_1_pair_2_one_ints;
  std::array<double, 4> xchang_0_pair_4_one_ints;
  std::array<double, 1> xchang_1_pair_3_one_ints;
};
*/
#endif  // GF2F12_H_

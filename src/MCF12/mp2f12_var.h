#ifndef MP2F12_H_
#define MP2F12_H_

#include <array>
#include <unordered_map>

#include "../qc_input.h"
#include "../basis/qc_basis.h"
#include "../electron_pair_list.h"
#include "../electron_list.h"
#include "F12_Traces.h"
#include "correlation_factors.h"

class MP2F12_V_Engine {
 public:
  explicit MP2F12_V_Engine(const IOPs& iops, const Basis& basis) :
      traces(basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2, iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::ELECTRONS])
  {
    correlation_factor = create_correlation_factor(iops);
    nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
    nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
    nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
  }
  ~MP2F12_V_Engine() {
    delete correlation_factor;
  }
  double calculate_v(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
 protected:
  //define the amplitudes
  static constexpr double a1 = 3.0/8.0;
  static constexpr double a2 = 1.0/8.0;

  // auto-generate the correct coefficients
  static constexpr double c1 = 2.0*a1-a2;
  static constexpr double c2 = 2.0*a2-a1;

  F12_Traces traces;
  Correlation_Factor* correlation_factor;
  double nsamp_pair;
  double nsamp_one_1;
  double nsamp_one_2;
};

class MP2F12_VBX_Engine : public MP2F12_V_Engine {
 public:
  explicit MP2F12_VBX_Engine(const IOPs& iops, const Basis& basis) : MP2F12_V_Engine(iops, basis) {
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

#endif  // MP2F12_H_

#ifndef MP2F12_H_
#define MP2F12_H_

#include <array>
#include <unordered_map>

#include "../qc_input.h"
#include "../basis/qc_basis.h"
#include "../electron_pair_list.h"
#include "../electron_list.h"
#include "../MCMP/mcmp.h"
#include "F12_Traces.h"
#include "correlation_factors.h"

class MP2_F12_V : public MCMP {
 public:
  explicit MP2_F12_V(const IOPs& iops, const Basis& basis, std::string extension="f12_V");
  ~MP2_F12_V();
  void energy(double& emp, std::vector<double>& control, OVPs& ovps, Electron_Pair_List* epl, Tau* tau) override {}
  void energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
 protected:
  void calculate_v(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
  std::array<double, 2> calculate_v_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
  std::array<double, 2> calculate_v_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);
  std::array<double, 2> calculate_v_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

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

class MP2_F12_VBX : public MP2_F12_V {
 public:
  explicit MP2_F12_VBX(const IOPs& iops, const Basis& basis);
  void energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

 protected:
  void zero();
  void calculate_bx(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

  void calculate_bx_t_fa_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fa_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fa_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fa(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);

  void calculate_bx_t_fb_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fb_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fb_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  double calculate_bx_t_fb_4e_help(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t size);
  void calculate_bx_t_fb(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);

  void calculate_bx_t_fc_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fc_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fc_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fc(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);

  void calculate_bx_t_fd_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fd_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fd_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_t_fd(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);


  void calculate_bx_k_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_k_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  void calculate_bx_k_5e(const Electron_Pair_List* electron_pair_list, const Electron_List* el_one_list);
  double calculate_bx_k_5e_help( const std::vector<double>&, const std::vector<double>& , const std::vector<double>&, const std::vector<double>& , const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t, size_t);
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

  std::vector<double> T_ip;
  std::vector<double> T_ip_io;
  std::vector<double> T_ip_jo;

  std::vector<double> T_io_jo;
  std::vector<double> T_io_ko;
  std::vector<double> T_jo_ko;
};
#endif  // MP2F12_H_

#ifndef GF2F12_H_
#define GF2F12_H_

#include <array>
#include <unordered_map>

#include "../qc_input.h"
#include "../basis/basis.h"
#include "electron_pair_list.h"
#include "electron_list.h"

#include "../MCGF/qc_mcgf.h"
#include "F12_Traces.h"
#include "x_traces.h"

#include "correlation_factor_data.h"

class GF2_F12_V : public MCGF {
 protected: 
  typedef Electron_Pair_List<std::vector, std::allocator> Electron_Pair_List_Type;
  typedef Electron_List<std::vector, std::allocator> Electron_List_Type;
  typedef Wavefunction<std::vector, std::allocator> Wavefunction_Type;
  typedef Correlation_Factor_Data<std::vector, std::allocator> Correlation_Factor_Data_Type;
  typedef F12_Traces<std::vector, std::allocator> F12_Traces_Type;

 public:
  explicit GF2_F12_V(IOPs& iops, std::string extension="f12_V");
  ~GF2_F12_V();
  void energy_f12(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction_Type>&,
     Electron_Pair_List_Type*, Electron_List_Type*);

 protected:
  double calculate_v_2e(Electron_Pair_List_Type* electron_pair_list, Electron_List_Type* electron_list);
  double calculate_v_3e(Electron_Pair_List_Type* electron_pair_list, Electron_List_Type* electron_list);
  double calculate_v_4e(Electron_Pair_List_Type* electron_pair_list, Electron_List_Type* electron_list);
  void calculate_v(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction_Type>&,
     Electron_Pair_List_Type*, Electron_List_Type*);

  void core(OVPS_Host& ovps, Electron_Pair_List_Type* electron_pair_list);
  void energy_no_diff(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction_Type>&,
     Electron_Pair_List_Type*, Tau*);
  void energy_diff(std::vector<std::vector<double>>&,
     std::unordered_map<int, Wavefunction_Type>&,
     Electron_Pair_List_Type*, Tau*);

  //define the amplitudes
  static constexpr double a1 = 3.0/8.0;
  static constexpr double a2 = 1.0/8.0;

  // auto-generate the correct coefficients
  static constexpr double c1 = a1-0.5*a2;
  static constexpr double c2 = a2-0.5*a1;

  F12_Traces_Type traces;
  X_Traces x_traces;
  std::shared_ptr<Correlation_Factor_Data_Type> correlation_factor;

  std::vector<double> core_11p;
  std::vector<double> core_12p;
  std::vector<double> core_22p;
  std::vector<double> core_13;
  std::vector<double> core_23;

  std::vector<double> T_ip_io;
  std::vector<double> T_ip_jo;
  std::vector<double> T_io_jo;

  double nsamp_pair;
  double nsamp_one_1;
  double nsamp_one_2;
};

class GF2_F12_VBX : public GF2_F12_V {
 public:
  explicit GF2_F12_VBX(IOPs& iops);
  void energy_f12(std::vector<std::vector<double>>&, 
     std::unordered_map<int, Wavefunction_Type>&,
     Electron_Pair_List_Type*, Electron_List_Type*);

 protected:
  void calculate_bx(std::vector<std::vector<double>>& egf, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fa_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fa_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fa_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fa(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  double calculate_bx_t_fb_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fb_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fb_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fb(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  double calculate_bx_t_fc_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fc_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fc_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fc(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  double calculate_bx_t_fd_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fd_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fd_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_t_fd(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  double calculate_bx_k_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_k_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_k_5e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
  double calculate_bx_k(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  static constexpr double c3 = (a1*a1+a2*a2-a1*a2);
  static constexpr double c4 = 0.5 * (4.0*a1*a2-a1*a1-a2*a2);

  double nsamp_one_3;
  double nsamp_one_4;

  // core arrays
  std::vector<double> core_11o;
  std::vector<double> core_12o;
  std::vector<double> core_d11p;
  std::vector<double> core_d12p;
  std::vector<double> core_d21p;
  std::vector<double> core_d22p;
  std::vector<double> core_d13;
  std::vector<double> core_d23;

  // scratch arrays
  std::vector<double> T_ip;
  std::vector<double> T_ip_ko;
  std::vector<double> T_io_ko;
  std::vector<double> T_io_lo;
  std::vector<double> T_jo_ko;
  std::vector<double> T_jo_lo;
  std::vector<double> T_ko_lo;
};
#endif  // GF2F12_H_

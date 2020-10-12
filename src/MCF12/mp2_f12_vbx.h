#ifndef MP2_F12_VBX_H_
#define MP2_F12_VBX_H_
#include "mp2_f12_v.h"

class MP2_F12_VBX : public MP2_F12_V<std::vector, std::allocator> {
 public:
  explicit MP2_F12_VBX(const IOPs& iops);
  void energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

 protected:
  void zero();
  void calculate_bx(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);

  double calculate_bx_t_fa_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fa_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fa_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fa(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);

  double calculate_bx_t_fb_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fb_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fb_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fb_4e_help_direct(const std::vector<double>&, const std::vector<double>&,
      const std::vector<double>&, const std::vector<double>&,
      const std::vector<double>&, const std::vector<double>&, 
      const std::vector<double>&, size_t size);
  double calculate_bx_t_fb_4e_help_exchange(const std::vector<double>&, const std::vector<double>&,
      const std::vector<double>&, const std::vector<double>&,
      const std::vector<double>&, const std::vector<double>&, 
      const std::vector<double>&, size_t size);
  double calculate_bx_t_fb(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);

  double calculate_bx_t_fc_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);

  double calculate_bx_t_fd_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);


  double calculate_bx_k_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void calculate_bx_k_4e_help(
      size_t electrons, size_t electron_pairs, double alpha,
      const std::vector<double>& S_ip_io, const std::vector<double>& S_ip_jo,
      const std::vector<double>& S_io_jo, std::vector<double>& T_io_jo,
      const std::vector<double>& S_jo, std::vector<double>& T_io);
  double calculate_bx_k_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_k_5e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_k_5e_direct_help( const std::vector<double>&, const std::vector<double>& , const std::vector<double>&, const std::vector<double>& , const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, size_t, size_t);
  double calculate_bx_k_5e_exchange_help();
  double calculate_bx_k   (const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void normalize();

  static constexpr double c3 = 2.0*(a1*a1+a2*a2-a1*a2);
  static constexpr double c4 = 4*a1*a2-a1*a1-a2*a2;

  double nsamp_one_3;
  double nsamp_one_4;

  std::vector<double> T_io_ko;
  std::vector<double> T_io_lo;
  std::vector<double> T_jo_ko;
  std::vector<double> T_jo_lo;
  std::vector<double> T_ko_lo;
};
#endif  // MP2_F12_VBX_H_

#ifndef MP2_F12_VBX_H_
#define MP2_F12_VBX_H_
#include "mp2_f12_v.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class MP2_F12_VBX : public MP2_F12_V<Container, Allocator> {
 protected:
  typedef Container<double, Allocator<double>> vector_double;
  typedef Blas_Wrapper<Container, Allocator> Blas_Wrapper_Type;
  typedef Correlation_Factor_Data<Container, Allocator> Correlation_Factor_Data_Type;
  typedef Electron_List<Container, Allocator> Electron_List_Type;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef F12_Traces<Container, Allocator> F12_Traces_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

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
  double calculate_bx_t_fb_4e_help_direct(const vector_double&, const vector_double&,
      const vector_double&, const vector_double&,
      const vector_double&, const vector_double&, 
      const vector_double&, size_t size);
  double calculate_bx_t_fb_4e_help_exchange(const vector_double&, const vector_double&,
      const vector_double&, const vector_double&,
      const vector_double&, const vector_double&, 
      const vector_double&, size_t size);
  double calculate_bx_t_fb(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);

  double calculate_bx_t_fc_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fc(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);

  double calculate_bx_t_fd_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  double calculate_bx_t_fd(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);


  void calculate_bx_k_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void calculate_bx_k_4e_help(
      size_t electrons, size_t electron_pairs, double alpha,
      const vector_double& S_ip_io, const vector_double& S_ip_jo,
      const vector_double& S_io_jo, vector_double& T_io_jo,
      const vector_double& S_jo);
  void calculate_bx_k_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void calculate_bx_k_5e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void calculate_bx_k_5e_direct_help(double alpha,
      const vector_double&, const vector_double&,
      const vector_double&, const vector_double&,
      const vector_double&, const vector_double&,
      const vector_double&, size_t, size_t);
  void calculate_bx_k_5e_exchange_help(
      double alpha,
      const vector_double& S_ip_io, const vector_double& S_ip_jo,
      const vector_double& S_ip_ko, const vector_double& S_io_jo,
      const vector_double& S_io_ko, const vector_double& S_jo_ko,
      const vector_double& weight, size_t electrons, size_t electron_pairs);
  double calculate_bx_k   (const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* el_one_list);
  void normalize();

  double nsamp_one_3;
  double nsamp_one_4;

  vector_double T_io_ko;
  vector_double T_io_lo;
  vector_double T_jo_ko;
  vector_double T_jo_lo;
  vector_double T_ko_lo;

 private:
  static constexpr double a1 = 3.0/8.0;
  static constexpr double a2 = 1.0/8.0;
  static constexpr double c1 = 2.0*a1-a2;
  static constexpr double c2 = 2.0*a2-a1;
  static constexpr double c3 = 2.0*(a1*a1+a2*a2-a1*a2);
  static constexpr double c4 = 4*a1*a2-a1*a1-a2*a2;
};

template class MP2_F12_VBX<std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class MP2_F12_VBX<thrust::device_vector, thrust::device_allocator>;
#endif
#endif  // MP2_F12_VBX_H_

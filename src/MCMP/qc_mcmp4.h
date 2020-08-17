#ifndef QC_MCMP4_H_
#define QC_MCMP4_H_

#include "mcmp.h"
#include "cblas.h"

template <int CVMP4>
class MCMP4 : public MCMP {
 public:
  MCMP4(Electron_Pair_List* electron_pair_list) : MCMP(CVMP4*(100 + CVMP4*(-135 + CVMP4*(68 - 9*CVMP4))) / 4, 3, "24", false),
      mpn(electron_pair_list->size()),
      rv(mpn),
      wgt(mpn),
      r_r(mpn),
      r_w(mpn),
      w_w(mpn),
      en_r(mpn),
      en_w(mpn),
      ij_(mpn * mpn),
      ik_(mpn * mpn),
      il_(mpn * mpn),
      jk_(mpn * mpn),
      jl_(mpn * mpn),
      kl_(mpn * mpn),
      ext_ptr(8),
      ext_data(8,
      std::vector<double>(mpn*mpn)),
      i_kl(mpn * mpn),
      j_kl(mpn * mpn),
      ij_rk(mpn * mpn),
      ij_wk(mpn * mpn),
      ij_rl(mpn * mpn),
      ij_wl(mpn * mpn),
      T_r(mpn * mpn),
      T_w(mpn * mpn),
      Av(mpn * mpn) {}
  void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List*, Tau*) override;
  void energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list){}

 private:
  void contract(std::vector<double>& result, const std::vector<double>& A, CBLAS_TRANSPOSE A_trans, const std::vector<double>& B, const std::vector<double>& v);
  void mcmp4_ij_helper(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ik, const std::vector<double>& jk,
      const std::vector<double>& il, const std::vector<double>& jl);
  void mcmp4_ij_helper_t1(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ik_1, const std::vector<double>& ik_2, const std::vector<double>& ik_3,
      const std::vector<double>& jk,
      const std::vector<double>& il,
      const std::vector<double>& jl_1, const std::vector<double>& jl_2, const std::vector<double>& jl_3);
  void mcmp4_ij_helper_t2(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ik_1, const std::vector<double>& ik_2,
      const std::vector<double>& jk_1, const std::vector<double>& jk_2,
      const std::vector<double>& il_1, const std::vector<double>& il_2,
      const std::vector<double>& jl_1, const std::vector<double>& jl_2);
  void mcmp4_ij_helper_t3(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ik,
      const std::vector<double>& jk_1, const std::vector<double>& jk_2, const std::vector<double>& jk_3,
      const std::vector<double>& il_1, const std::vector<double>& il_2, const std::vector<double>& il_3,
      const std::vector<double>& jl);
  void mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control, const OVPS_Host& ovps);
  void mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPS_Host& ovps);

  void mcmp4_il_helper(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ij, const std::vector<double>& jl,
      const std::vector<double>& ik, const std::vector<double>& kl);
  void mcmp4_il_helper_t1(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ij_1, const std::vector<double>& ij_2, const std::vector<double>& ij_3,
      const std::vector<double>& jl,
      const std::vector<double>& ik,
      const std::vector<double>& kl_1, const std::vector<double>& kl_2, const std::vector<double>& kl_3);
  void mcmp4_il_helper_t2(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ij_1, const std::vector<double>& ij_2,
      const std::vector<double>& jl_1, const std::vector<double>& jl_2,
      const std::vector<double>& ik_1, const std::vector<double>& ik_2,
      const std::vector<double>& kl_1, const std::vector<double>& kl_2);
  void mcmp4_il_helper_t3(double constant,
      double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& ij,
      const std::vector<double>& jl_1, const std::vector<double>& jl_2, const std::vector<double>& jl_3,
      const std::vector<double>& ik_1, const std::vector<double>& ik_2, const std::vector<double>& ik_3,
      const std::vector<double>& kl);
  void mcmp4_energy_il_fast(double& emp4, std::vector<double>& control, const OVPS_Host& ovps);

  std::array<double, 4> contract_jk(int walker,
      const std::vector<double>& T,
      const std::vector<double>& jk, const std::vector<double>& ik, const std::vector<double>& ij);
  void mcmp4_energy_ijkl_helper(double& emp4, std::vector<double>& control, int offset,
      const std::vector<double>& constants,
      const std::vector<const std::vector<double>*>& ij,
      const std::vector<const std::vector<double>*>& ik,
      const std::vector<double>& il,
      const std::vector<const std::vector<double>*>& jk,
      const std::vector<double>& jl,
      const std::vector<double>& kl);
  void mcmp4_energy_ijkl_t1(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const std::vector<double>*> ij,
      const std::vector<const std::vector<double>*> ik,
      const std::vector<double>& il_1, const std::vector<double>& il_2,
      const std::vector<const std::vector<double>*> jk_1, const std::vector<const std::vector<double>*> jk_2,
      const std::vector<double>& jl,
      const std::vector<double>& kl);
  void mcmp4_energy_ijkl_t2(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const std::vector<double>*> ij,
      const std::vector<const std::vector<double>*> ik_1, const std::vector<const std::vector<double>*> ik_2,
      const std::vector<double>& il,
      const std::vector<const std::vector<double>*> jk,
      const std::vector<double>& jl_1, const std::vector<double>& jl_2,
      const std::vector<double>& kl);
  void mcmp4_energy_ijkl_t3(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const std::vector<double>*> ij_1, const std::vector<const std::vector<double>*> ij_2,
      const std::vector<const std::vector<double>*> ik,
      const std::vector<double>& il,
      const std::vector<const std::vector<double>*> jk,
      const std::vector<double>& jl,
      const std::vector<double>& kl_1, const std::vector<double>& kl_2);
  void mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPS_Host& ovps);

  int mpn;
  std::vector<double> rv;
  std::vector<double> wgt;
  std::vector<double> r_r;
  std::vector<double> r_w;
  std::vector<double> w_w;
  std::vector<double> en_r;
  std::vector<double> en_w;

  std::vector<double> ij_;
  std::vector<double> ik_;
  std::vector<double> il_;
  std::vector<double> jk_;
  std::vector<double> jl_;
  std::vector<double> kl_;

  std::vector<const std::vector<double>*> ext_ptr;
  std::vector<std::vector<double>> ext_data;

  std::vector<double> i_kl;
  std::vector<double> j_kl;
  std::vector<double> ij_rk;
  std::vector<double> ij_wk;
  std::vector<double> ij_rl;
  std::vector<double> ij_wl;
  std::vector<double> T_r;
  std::vector<double> T_w;
  std::vector<double> Av;
};

template class MCMP4<0>;
template class MCMP4<1>;
template class MCMP4<2>;
template class MCMP4<3>;
template class MCMP4<4>;

MCMP* create_MCMP4(int cv_level, Electron_Pair_List* electron_pair_list);
#endif  // QC_MCMP4_H_

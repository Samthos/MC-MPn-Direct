//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"
#include "cblas.h"
class MP4_Engine {
 public:
  MP4_Engine(std::vector<el_pair_typ>& el_pair) :
      mpn(el_pair.size()),
      rv(mpn), wgt(mpn), r_r(mpn), r_w(mpn), en_r(mpn), en_w(mpn),
      ik_(mpn * mpn), jk_(mpn * mpn), il_(mpn * mpn), jl_(mpn * mpn),
      ij_ptr(8), ik_ptr(8), jk_ptr(8),
      ij_data(8, std::vector<double>(mpn*mpn)), ik_data(8, std::vector<double>(mpn*mpn)), jk_data(8, std::vector<double>(mpn*mpn)),
      i_kl(mpn * mpn), j_kl(mpn * mpn),
      ij_rk(mpn * mpn), ij_rl(mpn * mpn), ij_wl(mpn * mpn),
      T_r(mpn * mpn), T_w(mpn * mpn), Av(mpn * mpn) {
    std::transform(el_pair.begin(), el_pair.end(), rv.begin(), [](el_pair_typ ept){return ept.rv;});
    std::transform(el_pair.begin(), el_pair.end(), wgt.begin(), [](el_pair_typ ept){return 1.0/ept.wgt;});
    std::transform(rv.begin(), rv.end(), rv.begin(), r_r.begin(), std::multiplies<>());
    std::transform(rv.begin(), rv.end(), wgt.begin(), r_w.begin(), std::multiplies<>());
  }
  void energy(double &emp4, std::vector<double> &control, const OVPs &ovps) {
    //mcmp4_energy_ij_fast(emp4, control, ovps);
    //mcmp4_energy_ik_fast(emp4, control, ovps);
    mcmp4_energy_ijkl_fast(emp4, control, ovps);
  }
 private:
  void contract(std::vector<double>& result, const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& v);
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
  void mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);
  void mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);

  double contract_jk(int walker,
      const std::vector<double>& T, const std::vector<double>& rv,
      const std::vector<double>& jk, const std::vector<double>& ik, const std::vector<double>& ij);
  void mcmp4_energy_ijkl_helper(double& emp4, std::vector<double>& control,
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
  void mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);

  int mpn;
  std::vector<double> rv;
  std::vector<double> wgt;
  std::vector<double> r_r;
  std::vector<double> r_w;
  std::vector<double> en_r;
  std::vector<double> en_w;

  std::vector<double> ik_;
  std::vector<double> jk_;
  std::vector<double> il_;
  std::vector<double> jl_;

  std::vector<const std::vector<double>*> ij_ptr;
  std::vector<const std::vector<double>*> ik_ptr;
  std::vector<const std::vector<double>*> jk_ptr;
  std::vector<std::vector<double>> ij_data;
  std::vector<std::vector<double>> ik_data;
  std::vector<std::vector<double>> jk_data;

  std::vector<double> i_kl;
  std::vector<double> j_kl;
  std::vector<double> ij_rk;
  std::vector<double> ij_rl;
  std::vector<double> ij_wl;
  std::vector<double> T_r;
  std::vector<double> T_w;
  std::vector<double> Av;
};

void MP4_Engine::contract(std::vector<double>& result, const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& v) {
  Ddgmm(DDGMM_SIDE_LEFT,
      mpn, mpn,
      A.data(), mpn,
      v.data(), 1,
      Av.data(), mpn);
  cblas_dgemm(CblasColMajor,
      CblasTrans, CblasNoTrans,
      mpn, mpn, mpn,
      1.0,
      Av.data(), mpn,
      B.data(), mpn,
      0.0,
      result.data(), mpn);
}
void MP4_Engine::mcmp4_ij_helper(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ik, const std::vector<double>& jk,
    const std::vector<double>& il, const std::vector<double>& jl) {
  std::transform(ik.begin(), ik.end(), il.begin(), i_kl.begin(), std::multiplies<>());
  std::transform(jk.begin(), jk.end(), jl.begin(), j_kl.begin(), std::multiplies<>());

  contract(ij_rk, jk, ik, rv);

  contract(ij_rl, jl, il, rv);
  contract(ij_wl, jl, il, wgt);

  contract(T_r, j_kl, i_kl, r_r);
  contract(T_w, j_kl, i_kl, r_w);

  // combin ij_rk * ij_rl - T_r
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_rl), std::begin(ij_rl), std::multiplies<>());
  std::transform(std::begin(ij_rl), std::end(ij_rl), std::begin(T_r), std::begin(ij_rl), std::minus<>());

  // combin ij_rk * ij_wl - T_w
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_wl), std::begin(ij_wl), std::multiplies<>());
  std::transform(std::begin(ij_wl), std::end(ij_wl), std::begin(T_w), std::begin(ij_wl), std::minus<>());

  // zero diagonals
  cblas_dscal(mpn, 0.0, ij_rl.data(), mpn+1);
  cblas_dscal(mpn, 0.0, ij_wl.data(), mpn+1);

  // contract j
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mpn, mpn,
      1.0,
      ij_rl.data(), mpn,
      rv.data(), 1,
      0.0,
      en_r.data(), 1);
  // contract j
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mpn, mpn,
      1.0,
      ij_wl.data(), mpn,
      rv.data(), 1,
      0.0,
      en_w.data(), 1);
  emp4 += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;
  control[0 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant;
  control[1 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_w.begin(), 0.0) * constant;
  control[2 + offset] +=  std::inner_product(rv.begin(), rv.end(), en_w.begin(), 0.0) * constant;
}
void MP4_Engine::mcmp4_ij_helper_t1(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ik_1, const std::vector<double>& ik_2, const std::vector<double>& ik_3,
    const std::vector<double>& jk,
    const std::vector<double>& il,
    const std::vector<double>& jl_1, const std::vector<double>& jl_2, const std::vector<double>& jl_3
) {
  std::transform(ik_1.begin(), ik_1.end(), ik_2.begin(), ik_.begin(), std::multiplies<>());
  std::transform(ik_.begin(), ik_.end(), ik_3.begin(), ik_.begin(), std::multiplies<>());

  std::transform(jl_1.begin(), jl_1.end(), jl_2.begin(), jl_.begin(), std::multiplies<>());
  std::transform(jl_.begin(), jl_.end(), jl_3.begin(), jl_.begin(), std::multiplies<>());
  mcmp4_ij_helper(constant, emp4, control, offset, ik_, jk, il, jl_);
}
void MP4_Engine::mcmp4_ij_helper_t2(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ik_1, const std::vector<double>& ik_2,
    const std::vector<double>& jk_1, const std::vector<double>& jk_2,
    const std::vector<double>& il_1, const std::vector<double>& il_2,
    const std::vector<double>& jl_1, const std::vector<double>& jl_2
) {
  std::transform(ik_1.begin(), ik_1.end(), ik_2.begin(), ik_.begin(), std::multiplies<>());
  std::transform(jk_1.begin(), jk_1.end(), jk_2.begin(), jk_.begin(), std::multiplies<>());
  std::transform(il_1.begin(), il_1.end(), il_2.begin(), il_.begin(), std::multiplies<>());
  std::transform(jl_1.begin(), jl_1.end(), jl_2.begin(), jl_.begin(), std::multiplies<>());
  mcmp4_ij_helper(constant, emp4, control, offset, ik_, jk_, il_, jl_);
}
void MP4_Engine::mcmp4_ij_helper_t3(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ik,
    const std::vector<double>& jk_1, const std::vector<double>& jk_2, const std::vector<double>& jk_3,
    const std::vector<double>& il_1, const std::vector<double>& il_2, const std::vector<double>& il_3,
    const std::vector<double>& jl
) {
  std::transform(jk_1.begin(), jk_1.end(), jk_2.begin(), jk_.begin(), std::multiplies<>());
  std::transform(jk_.begin(), jk_.end(), jk_3.begin(), jk_.begin(), std::multiplies<>());

  std::transform(il_1.begin(), il_1.end(), il_2.begin(), il_.begin(), std::multiplies<>());
  std::transform(il_.begin(), il_.end(), il_3.begin(), il_.begin(), std::multiplies<>());
  mcmp4_ij_helper(constant, emp4, control, offset, ik, jk_, il_, jl);
}
void MP4_Engine::mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {
  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  */
  mcmp4_ij_helper_t1(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t1(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);

  mcmp4_ij_helper_t2(1, emp4, control, 0,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(1, emp4, control, 0,
      ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(-2, emp4, control, 0,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(-2, emp4, control, 0,
      ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(-16, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);

  mcmp4_ij_helper_t3(2, emp4, control, 0,
      ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(2, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(8, emp4, control, 0,
      ovps.v_set[1][0].s_11, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(8, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, 0,
      ovps.v_set[1][0].s_12, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, 0,
      ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, 0,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
}
void MP4_Engine::mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {
  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
      */
  mcmp4_ij_helper_t1(  4, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1(  4, emp4, control, 3,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1(  4, emp4, control, 3,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1( -2, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t1( -2, emp4, control, 3,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1( -8, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1( -8, emp4, control, 3,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1(  4, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);

  mcmp4_ij_helper_t2(  1, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2(  1, emp4, control, 3,
    ovps.v_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  2, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(  2, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  2, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(  2, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  8, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(  8, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(  8, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(  8, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2( -2, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2( -2, emp4, control, 3,
    ovps.v_set[0][0].s_22, ovps.v_set[0][0].s_11, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(-16, emp4, control, 3,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);

  mcmp4_ij_helper_t3(  4, emp4, control, 3,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, 3,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, 3,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3( -2, emp4, control, 3,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3( -2, emp4, control, 3,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3( -8, emp4, control, 3,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3( -8, emp4, control, 3,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
}

double MP4_Engine::contract_jk(
    int walker,
    const std::vector<double>& T, const std::vector<double>& rv,
    const std::vector<double>& jk, const std::vector<double>& ik, const std::vector<double>& ij) {
  std::transform(T.begin(), T.end(), jk.begin(), Av.begin(), std::multiplies<>());

  std::transform(rv.begin(), rv.end(), ik.begin() + walker * mpn, r_r.begin(), std::multiplies<>());
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mpn, mpn,
      1.0,
      Av.data(), mpn,
      r_r.data(), 1,
      0.0,
      r_w.data(), 1);

  std::transform(rv.begin(), rv.end(), ij.begin() + walker * mpn, r_r.begin(), std::multiplies<>());
  return std::inner_product(r_w.begin(), r_w.end(), r_r.begin(), 0.0);
}
void MP4_Engine::mcmp4_energy_ijkl_helper(double& emp4, std::vector<double>& control,
    const std::vector<double>& constants,
    const std::vector<const std::vector<double>*>& ij,
    const std::vector<const std::vector<double>*>& ik,
    const std::vector<double>& il,
    const std::vector<const std::vector<double>*>& jk,
    const std::vector<double>& jl,
    const std::vector<double>& kl
    ) {
  double en, ct;
  for (int i = 0; i < mpn; ++i) {
    std::transform(rv.begin(), rv.end(), il_.begin() + i * mpn, r_r.begin(), std::multiplies<>());
    contract(T_r, kl, jl, r_r);
    std::transform(wgt.begin(), wgt.end(), il_.begin() + i * mpn, r_r.begin(), std::multiplies<>());
    contract(T_w, kl, jl, r_r);

    en = 0.0;
    ct = 0.0;
    for (auto eqn = 0ull; eqn < constants.size(); ++eqn) {
      en += constants[eqn] * contract_jk(i, T_r, rv, *(jk[eqn]), *(ik[eqn]), *(ij[eqn]));
      ct += constants[eqn] * contract_jk(i, T_w, rv, *(jk[eqn]), *(ik[eqn]), *(ij[eqn]));
    }
    emp4 += en * rv[i];
    control[9] += en * wgt[i];
    control[10] += ct * rv[i];
    control[11] += ct * wgt[i];
  }
}
void MP4_Engine::mcmp4_energy_ijkl_t1(double& emp4, std::vector<double>& control,
      const std::vector<double> constants,
      const std::vector<const std::vector<double>*> ij,
      const std::vector<const std::vector<double>*> ik,
      const std::vector<double>& il_1, const std::vector<double>& il_2,
      const std::vector<const std::vector<double>*> jk_1, const std::vector<const std::vector<double>*> jk_2,
      const std::vector<double>& jl,
      const std::vector<double>& kl) {
  std::transform(il_1.begin(), il_1.end(), il_2.begin(), il_.begin(), std::multiplies<>());

  for (auto i = 0ull; i < jk_1.size(); ++i) {
    std::transform(jk_1[i]->begin(), jk_1[i]->end(), jk_2[i]->begin(), jk_data[i].begin(), std::multiplies<>());
    jk_ptr[i] = &jk_data[i];
  }

  mcmp4_energy_ijkl_helper(emp4, control, constants, ij, ik, il_, jk_ptr, jl, kl);
}
void MP4_Engine::mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {

  /*
  mcmp4_energy_ijkl_t1(emp4, control,
      { 4, -2, -2, 4, -8, 4, 4, -2},
      {&ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_12},
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_12},
      {&ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_21},
      ovps.v_set[2][1].s_12, ovps.v_set[2][2].s_11);
  */

  mcmp4_energy_ijkl_t1(emp4, control,
      { -2, -2, -2, 4, 4, 4, 4, -8},
      {&ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_11},
      ovps.o_set[2][0].s_11, ovps.o_set[2][0].s_22,
      {&ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22},
      {&ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_12, ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      {  2, 2, -2, -4, -4, 4},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_11},
      {&ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t1(emp4, control,
      { -2, 2, 2, 4, -4, -4},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11},
      {&ovps.o_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_12, ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      {  2, -2, -4, -4, 4, 8},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_21},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_11},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22},
      {&ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t1(emp4, control,
      { -2, 2, 4, -4, -4, 8},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_11},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_21},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21,
      {&ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22},
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_12, ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      { -4, -4, 4, 8, 8, -8},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_11},
      {&ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      {  4, -4, -4, -8, 8, 8},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12,
      {&ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11},
      {&ovps.o_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_11, ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      {  2, 2, -2, -4, -4, 4},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12},
      {&ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_11},
      {&ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      { -2, 2, 2, 4, -4, -4},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_11, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_12},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22,
      {&ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11},
      {&ovps.o_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_11, ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t1(emp4, control,
      { -2, -2, -2, 4, 4, 4, 4, -8},
      {&ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_21},
      {&ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11},
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22,
      {&ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_22},
      {&ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.o_set[2][2].s_11);
}

void MP::mcmp4_energy(double& emp4, std::vector<double>& control) {
  MP4_Engine mp4(el_pair_list);
  emp4 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  mp4.energy(emp4, control, ovps);
  printf("%12.6f : ", emp4);
  for (const auto &item : control) {
    printf("%12.6f", item);
  }
  printf("\n");
  emp4 = 0;
  std::fill(control.begin(), control.end(), 0.0);

  //mcmp4_energy_ij(emp4, control);
  //mcmp4_energy_ik(emp4, control);
  //mcmp4_energy_il(emp4, control);
  mcmp4_energy_ijkl(emp4, control);
  printf("%12.6f : ", emp4);
  for (const auto &item : control) {
    printf("%12.6f", item);
  }
  printf("\n");

  auto tau_wgt = tau.get_wgt(3);
  std::transform(control.begin(), control.end(), control.begin(),
                 [&](double x) { return x * tau_wgt; });
  emp4 *= tau_wgt;

  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 3);
  emp4 /= nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}

void MP::mcmp4_energy_ij(double& emp4, std::vector<double>& control) {
  // ij contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

        #include "qc_mcmp4_ij.h"

        en_corr += en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr += en * el_pair_list[kt].rv / el_pair_list[lt].wgt;
      }
      //printf("%12.6f", (en_kt[0] * ct_lt[0] - ct_corr) * 1000000);

      en_i += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -en_corr) * el_pair_list[jt].rv;
      ct_i += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -ct_corr) * el_pair_list[jt].rv;
    }
    //printf("\n");
    emp4 += en_i * el_pair_list[it].rv;
    control[0] += en_i / el_pair_list[it].wgt;
    control[1] += ct_i / el_pair_list[it].wgt;
    control[2] += ct_i * el_pair_list[it].rv;
  }
}
void MP::mcmp4_energy_ik(double& emp4, std::vector<double>& control) {
  // ik contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
      if (it == kt) continue;
      auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_j.h"
        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list[jt].rv;});
        // std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_l.h"
        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto jt = 0, lt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++, lt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_ik.h"
        en_corr += en * el_pair_list[jt].rv * el_pair_list[lt].rv;
        ct_corr += en * el_pair_list[jt].rv / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -en_corr) * el_pair_list[kt].rv;
      ct_i += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -ct_corr) * el_pair_list[kt].rv;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[3] += en_i / el_pair_list[it].wgt;
    control[4] += ct_i / el_pair_list[it].wgt;
    control[5] += ct_i * el_pair_list[it].rv;
  }
}
void MP::mcmp4_energy_il(double& emp4, std::vector<double>& control) {
  // il contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
      if (it == lt) continue;
      auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_il_k.h"
        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        // std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_il_j.h"
        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list[jt].rv;});
        // std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, jt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, jt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_il.h"
        en_corr += en * el_pair_list[kt].rv * el_pair_list[jt].rv;
        // ct_corr += en / el_pair_list[kt].wgt / el_pair_list[jt].wgt;
      }

      en_i += std::inner_product(en_kt.begin(), en_kt.end(), en_jt.begin(), -en_corr) * el_pair_list[lt].rv;
      ct_i += std::inner_product(en_kt.begin(), en_kt.end(), en_jt.begin(), -en_corr) / el_pair_list[lt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[6] += en_i / el_pair_list[it].wgt;
    control[7] += ct_i / el_pair_list[it].wgt;
    control[8] += ct_i * el_pair_list[it].rv;
  }
}
void MP::mcmp4_energy_ijkl(double& emp4, std::vector<double>& control) {
  // fourth order sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_jkl = 0;
    double ct_jkl = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      double en_kl = 0;
      double ct_kl = 0;
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 30> en_l;
        std::array<double, 30> ct_l;
        en_l.fill(0.0);
        ct_l.fill(0.0);
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          std::array<double, 30> en;

          #include "qc_mcmp4_ijkl.h"

          std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y){return x + y * el_pair_list[lt].rv;});
          std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y){return x + y / el_pair_list[lt].wgt;});
        }
        double en_l_t = 0;
        double ct_l_t = 0;
#include "qc_mcmp4_ijk.h"
        en_kl += en_l_t * el_pair_list[kt].rv;
        ct_kl += ct_l_t * el_pair_list[kt].rv;
      }
      en_jkl += en_kl * el_pair_list[jt].rv;
      ct_jkl += ct_kl * el_pair_list[jt].rv;
    }
    emp4       += en_jkl * el_pair_list[it].rv;

    control[9] += en_jkl / el_pair_list[it].wgt;
    control[10] += ct_jkl * el_pair_list[it].rv;
    control[11] += ct_jkl / el_pair_list[it].wgt;
  }
}


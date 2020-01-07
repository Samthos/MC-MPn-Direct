//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"
#include "cblas.h"
class MP4_Engine {
 public:
  MP4_Engine(Electron_Pair_List* el_pair) :
      mpn(el_pair->size()),
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
      Av(mpn * mpn) {
    std::transform(el_pair->begin(), el_pair->end(), rv.begin(), [](Electron_Pair ept){return ept.rv;});
    std::transform(el_pair->begin(), el_pair->end(), wgt.begin(), [](Electron_Pair ept){return 1.0/ept.wgt;});

    std::transform(rv.begin(), rv.end(), rv.begin(), r_r.begin(), std::multiplies<>());
    std::transform(rv.begin(), rv.end(), wgt.begin(), r_w.begin(), std::multiplies<>());
    std::transform(wgt.begin(), wgt.end(), wgt.begin(), w_w.begin(), std::multiplies<>());
  }
  void energy(double &emp4, std::vector<double> &control, const OVPs &ovps) {
    mcmp4_energy_ij_fast(emp4, control, ovps);
    mcmp4_energy_ik_fast(emp4, control, ovps);
    mcmp4_energy_il_fast(emp4, control, ovps);
    mcmp4_energy_ijkl_fast(emp4, control, ovps);
  }
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
  void mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);
  void mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);

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
  void mcmp4_energy_il_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);

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
  void mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPs& ovps);

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

void MP4_Engine::contract(std::vector<double>& result, const std::vector<double>& A, CBLAS_TRANSPOSE A_trans, const std::vector<double>& B, const std::vector<double>& v) {
  Ddgmm(DDGMM_SIDE_LEFT,
      mpn, mpn,
      B.data(), mpn,
      v.data(), 1,
      Av.data(), mpn);
  cblas_dgemm(CblasColMajor,
      A_trans, CblasNoTrans,
      mpn, mpn, mpn,
      1.0,
      A.data(), mpn,
      Av.data(), mpn,
      0.0,
      result.data(), mpn);
}
void MP4_Engine::mcmp4_ij_helper(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ik, const std::vector<double>& jk,
    const std::vector<double>& il, const std::vector<double>& jl) {

  // ij_rk = ik . diagonal( rv ) . kj
  contract(ij_rk, jk, CblasTrans, ik, rv);
  contract(ij_rl, jl, CblasTrans, il, rv);

  // ij_rl = il . diagonal( rv ) . lj
#if MP4CV >= 3
  contract(ij_wk, jk, CblasTrans, ik, wgt);
  contract(ij_wl, jl, CblasTrans, il, wgt);
#endif

  // i_kl = ik * il
  std::transform(ik.begin(), ik.end(), il.begin(), i_kl.begin(), std::multiplies<>());
  std::transform(jk.begin(), jk.end(), jl.begin(), j_kl.begin(), std::multiplies<>());

  // T_r = i_kl . diagonal(rv * rv) . kl_j
  contract(T_r, j_kl, CblasTrans, i_kl, r_r);

  // ij_WlWk = ij_rk * ij_rl - T_r
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_rl), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  emp4 += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;                   // r r r r
#if MP4CV >= 1
  control[0 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r r r
#endif

#if MP4CV >= 2
  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[1 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w r r
#endif

#if MP4CV >= 3
  // T_r = i_kl . diagonal(wgt * rv) . kl_j
  contract(T_r, j_kl, CblasTrans, i_kl, r_w);

  // T_w = ij_rk * ij_wl - T_r
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_wl), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[2 + offset] += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;    // r r r w
  control[3 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r r w

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[4 + offset] += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;    // r w r w
  control[5 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w r w

  // T_w = ij_rk * ij_wl - T_r
  std::transform(std::begin(ij_wk), std::end(ij_wk), std::begin(ij_rl), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[6 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r w r

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[7 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w w r

  // T_r = i_kl . diagonal(wgt * wgt) . kl_j
  contract(T_r, j_kl, CblasTrans, i_kl, w_w);

  // T_w = ij_wk * ij_wl - T_r
  std::transform(std::begin(ij_wk), std::end(ij_wk), std::begin(ij_wl), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[8 + offset] += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;    // r r w w
  control[9 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r w w

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[10 + offset] += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;   // r w w w
  control[11 + offset] += std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w w w
#endif
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
  constexpr int offset = 0;
  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  */
  mcmp4_ij_helper_t1(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t1(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);

  mcmp4_ij_helper_t2(1, emp4, control, offset,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(1, emp4, control, offset,
      ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(-2, emp4, control, offset,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(-2, emp4, control, offset,
      ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(-16, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);

  mcmp4_ij_helper_t3(2, emp4, control, offset,
      ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(2, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(8, emp4, control, offset,
      ovps.v_set[1][0].s_11, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(8, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, offset,
      ovps.v_set[1][0].s_12, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, offset,
      ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(-4, emp4, control, offset,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
}
void MP4_Engine::mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {
  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
      */
  constexpr int offset = 12;
  mcmp4_ij_helper_t1(  4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1(  4, emp4, control, offset,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1(  4, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1( -2, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t1( -2, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1( -8, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1( -8, emp4, control, offset,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1(  4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);

  mcmp4_ij_helper_t2(  1, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2(  1, emp4, control, offset,
    ovps.v_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  2, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(  2, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  2, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(  2, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(  8, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(  8, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(  8, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(  8, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2( -2, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2( -2, emp4, control, offset,
    ovps.v_set[0][0].s_22, ovps.v_set[0][0].s_11, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2( -4, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(-16, emp4, control, offset,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);

  mcmp4_ij_helper_t3(  4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3(  4, emp4, control, offset,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3( -2, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3( -2, emp4, control, offset,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3( -8, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3( -8, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
}

void MP4_Engine::mcmp4_il_helper(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ij, const std::vector<double>& jl,
    const std::vector<double>& ik, const std::vector<double>& kl) {
  std::transform(ij.begin(), ij.end(), ik.begin(), i_kl.begin(), std::multiplies<>());
  std::transform(jl.begin(), jl.end(), kl.begin(), j_kl.begin(), std::multiplies<>());

  // contract k and j
  contract(ij_rl, jl, CblasNoTrans, ij, rv);
  contract(ij_rk, kl, CblasNoTrans, ik, rv);

#if MP4CV >= 3
  contract(ij_wl, jl, CblasNoTrans, ij, wgt);
  contract(ij_wk, kl, CblasNoTrans, ik, wgt);
#endif

  // build jrkr
  contract(T_r, j_kl, CblasNoTrans, i_kl, r_r);
  // combin ij_rk * ij_rl - T_r
  std::transform(std::begin(ij_rl), std::end(ij_rl), std::begin(ij_rk), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  emp4 += std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant; // r r r r
#if MP4CV >= 1
  control[0 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r r r
#endif

#if MP4CV >= 2
  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[1 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r r w
  control[2 + offset] +=  std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;   // r r r w
#endif


#if MP4CV >= 3
  contract(T_r, j_kl, CblasNoTrans, i_kl, r_w);
  // build jw kr
  std::transform(std::begin(ij_wl), std::end(ij_wl), std::begin(ij_rk), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[3 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w r r

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[4 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w r w
  control[5 + offset] +=  std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;   // r w r w

  // build jr kw
  std::transform(std::begin(ij_rl), std::end(ij_rl), std::begin(ij_wk), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[6 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r w r

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[7 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w r w w
  control[8 + offset] +=  std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;   // r r w w

  // build jw kw
  contract(T_r, j_kl, CblasNoTrans, i_kl, w_w);
  std::transform(std::begin(ij_wl), std::end(ij_wl), std::begin(ij_wk), std::begin(T_w), std::multiplies<>());
  std::transform(std::begin(T_w), std::end(T_w), std::begin(T_r), std::begin(T_w), std::minus<>());
  cblas_dscal(mpn, 0.0, T_w.data(), mpn+1);

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, rv.data(), 1, 0.0, en_r.data(), 1);
  control[9 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w w r

  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, T_w.data(), mpn, wgt.data(), 1, 0.0, en_r.data(), 1);
  control[10 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), en_r.begin(), 0.0) * constant; // w w w w
  control[11 + offset] +=  std::inner_product(rv.begin(), rv.end(), en_r.begin(), 0.0) * constant;   // r w w w
#endif
}
void MP4_Engine::mcmp4_il_helper_t1(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ij_1, const std::vector<double>& ij_2, const std::vector<double>& ij_3,
    const std::vector<double>& jl,
    const std::vector<double>& ik,
    const std::vector<double>& kl_1, const std::vector<double>& kl_2, const std::vector<double>& kl_3
) {
  std::transform(ij_1.begin(), ij_1.end(), ij_2.begin(), ij_.begin(), std::multiplies<>());
  std::transform(ij_3.begin(), ij_3.end(), ij_.begin(), ij_.begin(), std::multiplies<>());
  std::transform(kl_1.begin(), kl_1.end(), kl_2.begin(), kl_.begin(), std::multiplies<>());
  std::transform(kl_3.begin(), kl_3.end(), kl_.begin(), kl_.begin(), std::multiplies<>());
  mcmp4_il_helper(constant, emp4, control, offset, ij_, jl, ik, kl_);
}
void MP4_Engine::mcmp4_il_helper_t2(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ij_1, const std::vector<double>& ij_2,
    const std::vector<double>& jl_1, const std::vector<double>& jl_2,
    const std::vector<double>& ik_1, const std::vector<double>& ik_2,
    const std::vector<double>& kl_1, const std::vector<double>& kl_2
) {
  std::transform(ij_1.begin(), ij_1.end(), ij_2.begin(), ij_.begin(), std::multiplies<>());
  std::transform(jl_1.begin(), jl_1.end(), jl_2.begin(), jl_.begin(), std::multiplies<>());
  std::transform(ik_1.begin(), ik_1.end(), ik_2.begin(), ik_.begin(), std::multiplies<>());
  std::transform(kl_1.begin(), kl_1.end(), kl_2.begin(), kl_.begin(), std::multiplies<>());
  mcmp4_il_helper(constant, emp4, control, offset, ij_, jl_, ik_, kl_);
}
void MP4_Engine::mcmp4_il_helper_t3(double constant,
    double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& ij,
    const std::vector<double>& jl_1, const std::vector<double>& jl_2, const std::vector<double>& jl_3,
    const std::vector<double>& ik_1, const std::vector<double>& ik_2, const std::vector<double>& ik_3,
    const std::vector<double>& kl) {
  std::transform(jl_1.begin(), jl_1.end(), jl_2.begin(), jl_.begin(), std::multiplies<>());
  std::transform(jl_3.begin(), jl_3.end(), jl_.begin(), jl_.begin(), std::multiplies<>());
  std::transform(ik_1.begin(), ik_1.end(), ik_2.begin(), ik_.begin(), std::multiplies<>());
  std::transform(ik_3.begin(), ik_3.end(), ik_.begin(), ik_.begin(), std::multiplies<>());
  mcmp4_il_helper(constant, emp4, control, offset, ij, jl_, ik_, kl);
}
void MP4_Engine::mcmp4_energy_il_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {
  constexpr int offset = 24;
  mcmp4_il_helper_t1( 2, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12,
      ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t1( 2, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t1( 8, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12,
      ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t1( 8, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t1( -4, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12,
      ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t1( -4, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12,
      ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t1( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t1( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);

  mcmp4_il_helper_t2( 1, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22,
      ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( 1, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_il_helper_t2( 2, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t2( 2, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( 2, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t2( 2, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( 8, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t2( 8, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t2( 8, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t2( 8, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t2( -2, emp4, control, offset,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22,
      ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( -2, emp4, control, offset,
      ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_il_helper_t2( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_il_helper_t2(-16, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);

  mcmp4_il_helper_t3( 2, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t3( 2, emp4, control, offset,
      ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_il_helper_t3( 8, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t3( 8, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11);
  mcmp4_il_helper_t3( -4, emp4, control, offset,
      ovps.o_set[0][0].s_21, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t3( -4, emp4, control, offset,
      ovps.o_set[0][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12,
      ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[2][2].s_11);
  mcmp4_il_helper_t3( -4, emp4, control, offset,
      ovps.v_set[0][0].s_12, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.o_set[2][2].s_11);
  mcmp4_il_helper_t3( -4, emp4, control, offset,
      ovps.v_set[0][0].s_22, ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.o_set[2][2].s_11);
}

std::array<double, 4> MP4_Engine::contract_jk(
    int walker,
    const std::vector<double>& T, 
    const std::vector<double>& jk, const std::vector<double>& ik, const std::vector<double>& ij) {
  std::array<double, 4> out;
  std::transform(T.begin(), T.end(), jk.begin(), Av.begin(), std::multiplies<>());

  std::transform(rv.begin(), rv.end(), ik.begin() + walker * mpn, ij_rk.begin(), std::multiplies<>());
  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, Av.data(), mpn, ij_rk.data(), 1, 0.0, en_r.data(), 1); // kr ?l

  std::transform(rv.begin(), rv.end(), ij.begin() + walker * mpn, ij_rk.begin(), std::multiplies<>());
  out[0] = std::inner_product(en_r.begin(), en_r.end(), ij_rk.begin(), 0.0); // jr kr l?
#if MP4CV >= 2
  std::transform(wgt.begin(), wgt.end(), ik.begin() + walker * mpn, ij_wk.begin(), std::multiplies<>());
  cblas_dgemv(CblasColMajor, CblasTrans, mpn, mpn, 1.0, Av.data(), mpn, ij_wk.data(), 1, 0.0, en_w.data(), 1); // kw ?l
  out[1] = std::inner_product(en_w.begin(), en_w.end(), ij_rk.begin(), 0.0); // jr kw l?
#endif

#if MP4CV >= 3
  std::transform(wgt.begin(), wgt.end(), ij.begin() + walker * mpn, ij_wk.begin(), std::multiplies<>());
  out[2] = std::inner_product(en_r.begin(), en_r.end(), ij_wk.begin(), 0.0); // jw kr l?
  out[3] = std::inner_product(en_w.begin(), en_w.end(), ij_wk.begin(), 0.0); // jw kw l?
#endif

  return out;
}
void MP4_Engine::mcmp4_energy_ijkl_helper(double& emp4, std::vector<double>& control, int offset,
    const std::vector<double>& constants,
    const std::vector<const std::vector<double>*>& ij,
    const std::vector<const std::vector<double>*>& ik,
    const std::vector<double>& il,
    const std::vector<const std::vector<double>*>& jk,
    const std::vector<double>& jl,
    const std::vector<double>& kl) {
  std::array<double, 4> contracted_jk;
  double jr_kr_lr, jr_kr_lw, jr_kw_lr, jr_kw_lw, jw_kr_lr, jw_kr_lw, jw_kw_lr, jw_kw_lw;
  for (int i = 0; i < mpn; ++i) {
    std::transform(rv.begin(), rv.end(), il.begin() + i * mpn, en_r.begin(), std::multiplies<>());
    contract(T_r, kl, CblasTrans, jl, en_r);

#if MP4CV >= 4
    std::transform(wgt.begin(), wgt.end(), il.begin() + i * mpn, en_w.begin(), std::multiplies<>());
    contract(T_w, kl, CblasTrans, jl, en_w);
#endif

    jr_kr_lr = 0; jr_kr_lw = 0; jr_kw_lr = 0; jr_kw_lw = 0;
    jw_kr_lr = 0; jw_kr_lw = 0; jw_kw_lr = 0; jw_kw_lw = 0;
    for (auto eqn = 0ull; eqn < constants.size(); ++eqn) {
      contracted_jk = contract_jk(i, T_r, *(jk[eqn]), *(ik[eqn]), *(ij[eqn]));
      jr_kr_lr += constants[eqn] * contracted_jk[0] ;
#if MP4CV >= 2
      jr_kw_lr += constants[eqn] * contracted_jk[1];
#endif
#if MP4CV >= 3
      jw_kr_lr += constants[eqn] * contracted_jk[2];
      jw_kw_lr += constants[eqn] * contracted_jk[3];
#endif

#if MP4CV >= 4
      contracted_jk = contract_jk(i, T_w, *(jk[eqn]), *(ik[eqn]), *(ij[eqn]));
      jr_kr_lw += constants[eqn] * contracted_jk[0];
      jr_kw_lw += constants[eqn] * contracted_jk[1];
      jw_kr_lw += constants[eqn] * contracted_jk[2];
      jw_kw_lw += constants[eqn] * contracted_jk[3];
#endif
    }
    emp4        += jr_kr_lr * rv[i];
#if MP4CV >= 1
    control[offset + 4] += jr_kr_lr * wgt[i];
#endif
#if MP4CV >= 2
    control[offset + 6] += jr_kw_lr * wgt[i];
#endif
#if MP4CV >= 3
    control[offset + 8] += jw_kr_lr * wgt[i];
    control[offset + 10] += jw_kw_lr * wgt[i];
#endif
#if MP4CV >= 4
    control[offset + 0] += jr_kr_lw * rv[i];
    control[offset + 1] += jw_kr_lw * rv[i];
    control[offset + 2] += jr_kw_lw * rv[i];
    control[offset + 3] += jw_kw_lw * rv[i];
    control[offset + 5] += jr_kr_lw * wgt[i];
    control[offset + 7] += jw_kr_lw * wgt[i];
    control[offset + 9] += jr_kw_lw * wgt[i];
    control[offset + 11] += jw_kw_lw * wgt[i];
#endif
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
    std::transform(jk_1[i]->begin(), jk_1[i]->end(), jk_2[i]->begin(), ext_data[i].begin(), std::multiplies<>());
    ext_ptr[i] = &ext_data[i];
  }

  mcmp4_energy_ijkl_helper(emp4, control, 36, constants, ij, ik, il_, ext_ptr, jl, kl);
}
void MP4_Engine::mcmp4_energy_ijkl_t2(double& emp4, std::vector<double>& control,
    const std::vector<double> constants,
    const std::vector<const std::vector<double>*> ij,
    const std::vector<const std::vector<double>*> ik_1, const std::vector<const std::vector<double>*> ik_2,
    const std::vector<double>& il,
    const std::vector<const std::vector<double>*> jk,
    const std::vector<double>& jl_1, const std::vector<double>& jl_2,
    const std::vector<double>& kl) {

  for (auto i = 0ull; i < ik_1.size(); ++i) {
    std::transform(ik_1[i]->begin(), ik_1[i]->end(), ik_2[i]->begin(), ext_data[i].begin(), std::multiplies<>());
    ext_ptr[i] = &ext_data[i];
  }

  std::transform(jl_1.begin(), jl_1.end(), jl_2.begin(), jl_.begin(), std::multiplies<>());

  mcmp4_energy_ijkl_helper(emp4, control, 48, constants, ij, ext_ptr, il, jk, jl_, kl);
}
void MP4_Engine::mcmp4_energy_ijkl_t3(double& emp4, std::vector<double>& control,
    const std::vector<double> constants,
    const std::vector<const std::vector<double>*> ij_1, const std::vector<const std::vector<double>*> ij_2,
    const std::vector<const std::vector<double>*> ik,
    const std::vector<double>& il,
    const std::vector<const std::vector<double>*> jk,
    const std::vector<double>& jl,
    const std::vector<double>& kl_1, const std::vector<double>& kl_2) {
  for (auto i = 0ull; i < ij_1.size(); ++i) {
    std::transform(ij_1[i]->begin(), ij_1[i]->end(), ij_2[i]->begin(), ext_data[i].begin(), std::multiplies<>());
    ext_ptr[i] = &ext_data[i];
  }

  std::transform(kl_1.begin(), kl_1.end(), kl_2.begin(), jl_.begin(), std::multiplies<>());

  mcmp4_energy_ijkl_helper(emp4, control, 60, constants, ext_ptr, ik, il, jk, jl, jl_);
}
void MP4_Engine::mcmp4_energy_ijkl_fast(double& emp4, std::vector<double>& control, const OVPs& ovps) {
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

  mcmp4_energy_ijkl_t2(emp4, control,
      {2, 2, 2, -4, -4, -4, -4, 8},
      {&ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_12},
      {&ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22},
      ovps.v_set[2][0].s_12 ,
      {&ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_11},
      ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22,
      ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {-4, 4, 4, 8, -8, -8},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11,
      ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t2(emp4, control,
      {-2, -2, 2, 4, 4, -4},
      {&ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.v_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_11,
      ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {2, -2, -2, -4, 4, 4},
      {&ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_11, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_12},
      ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21,
      ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t2(emp4, control,
      {-2, 2, 4, 4, -4, -8},
      {&ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_11},
      {&ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22},
      ovps.v_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_21},
      ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_21,
      ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {2, -2, -2, -4, 4, 4},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11},
      ovps.o_set[2][0].s_12 ,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12,
      ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {4, 4, -4, -8, -8, 8},
      {&ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_12,
      ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {2, -2, -4, 4, 4, -8},
      {&ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_12},
      {&ovps.v_set[1][0].s_11, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_11},
      ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {-2, -2, 2, 4, 4, -4},
      {&ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_21, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12},
      ovps.o_set[2][1].s_12, ovps.v_set[2][1].s_22,
      ovps.o_set[2][2].s_11);
  mcmp4_energy_ijkl_t2(emp4, control,
      {2, 2, 2, -4, -4, -4, -4, 8},
      {&ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_12},
      {&ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      {&ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_22},
      ovps.o_set[2][0].s_12 ,
      {&ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_11},
      ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22,
      ovps.o_set[2][2].s_11);


  mcmp4_energy_ijkl_t3(emp4, control,
      { -2, -2, -2, 4, 4, 4, 4, -8},
      {&ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22},
      {&ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_11},
      ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.o_set[1][1].s_22, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_12},
      ovps.v_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_energy_ijkl_t3(emp4, control,
      { -4, -4, 4, 8, 8, -8},
      {&ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_11},
      {&ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t3(emp4, control,
      { -4, -8, 8, -4, 4, 8},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11},
      {&ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.v_set[2][0].s_12,
      {&ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 2, 2, -2, -4, -4, 4},
      {&ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_11},
      {&ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 2, 4, -4, 2, -2, -4},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11},
      {&ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_11, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21},
      ovps.v_set[2][0].s_12,
      {&ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 2, 2, -2, -4, -4, 4},
      {&ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_11},
      {&ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.v_set[2][1].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 2, 4, -4, 2, -2, -4},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_11},
      {&ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12, &ovps.v_set[0][0].s_22},
      {&ovps.o_set[1][0].s_12, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_12, &ovps.o_set[1][0].s_22, &ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_22},
      ovps.v_set[2][0].s_11,
      {&ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_22, &ovps.v_set[1][1].s_22},
      ovps.o_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 2, -2, -4, -4, 4, 8},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_11, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.o_set[0][0].s_22},
      {&ovps.v_set[0][0].s_11, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_21, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_12, &ovps.o_set[1][0].s_21, &ovps.v_set[1][0].s_12},
      ovps.o_set[2][0].s_12,
      {&ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_11, &ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.o_set[1][1].s_11},
      ovps.v_set[2][1].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_energy_ijkl_t3(emp4, control,
      { -2, -4, 8, 2, -4, 4},
      {&ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_11, &ovps.o_set[0][0].s_12, &ovps.o_set[0][0].s_21},
      {&ovps.o_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.o_set[0][0].s_12},
      {&ovps.v_set[1][0].s_22, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_11, &ovps.o_set[1][0].s_21, &ovps.o_set[1][0].s_21, &ovps.v_set[1][0].s_22},
      ovps.v_set[2][0].s_11,
      {&ovps.o_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_12, &ovps.o_set[1][1].s_21},
      ovps.o_set[2][1].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_energy_ijkl_t3(emp4, control,
      { 4, -2, -2, 4, -2, 4, 4, -8},
      {&ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_21, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22, &ovps.o_set[0][0].s_22},
      {&ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_12, &ovps.v_set[0][0].s_22, &ovps.v_set[0][0].s_22},
      {&ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_12, &ovps.v_set[1][0].s_21, &ovps.v_set[1][0].s_22, &ovps.v_set[1][0].s_11, &ovps.v_set[1][0].s_12},
      ovps.o_set[2][0].s_12,
      {&ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_22, &ovps.v_set[1][1].s_21, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_11, &ovps.v_set[1][1].s_12, &ovps.v_set[1][1].s_11},
      ovps.o_set[2][1].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
}

void MP::mcmp4_energy(double& emp4, std::vector<double>& control4) {
  MP4_Engine mp4(el_pair_list);
  double en4 = 0.0;
  std::vector<double> ctrl(control4.size(), 0.0);

  mp4.energy(en4, ctrl, ovps);

  /*
  double emp4_debug = 0.0;
  std::vector<double> control_debug(control.size(), 0.0);
  mcmp4_energy_ij(emp4_debug, control_debug);
  mcmp4_energy_ik(emp4_debug, control_debug);
  mcmp4_energy_il(emp4_debug, control_debug);
  mcmp4_energy_ijkl(emp4_debug, control_debug);
  printf("xxx: %14.6f %14.6f\n", emp4, emp4_debug);
  int iter = 0;
  for (auto control_it = control.begin(), debug_it = control_debug.begin(); control_it != control.end(); control_it++, debug_it++) {
    printf("xxx: %2i %14.6f %14.6f\n", iter++, *control_it, *debug_it);
  }
  */

  auto nsamp_tauwgt = tau->get_wgt(3);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 3);
  emp4 = emp4 + en4 * nsamp_tauwgt;
#if MP4CV >= 1
  std::transform(ctrl.begin(), ctrl.end(), control4.begin(), control4.begin(), [&](double c, double total) { return total + c * nsamp_tauwgt; });
#endif
}

/*
WARNING:
WARNING: Don't delete the four following function.
WARNING: They are a much slower reference implementation of MC-MP4
WARNING: The MP4 class and the following should produce the exact same output 
WARNING:
*/
void MP::mcmp4_energy_ij(double& emp4, std::vector<double>& control) {
  // ij contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
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

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(kt).rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(kt).wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(lt).rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(lt).wgt;});
      }

      double kr_lr_corr = 0;
      double kr_lw_corr = 0;
      double kw_lr_corr = 0;
      double kw_lw_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

        #include "qc_mcmp4_ij.h"

        kr_lr_corr += en * el_pair_list->get(kt).rv * el_pair_list->get(lt).rv;
        kr_lw_corr += en * el_pair_list->get(kt).rv / el_pair_list->get(lt).wgt;
        kw_lr_corr += en / el_pair_list->get(kt).wgt * el_pair_list->get(lt).rv;
        kw_lw_corr += en / el_pair_list->get(kt).wgt / el_pair_list->get(lt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -kr_lr_corr) * el_pair_list->get(jt).rv;
      jr_kw_lr_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), en_lt.begin(), -kw_lr_corr) * el_pair_list->get(jt).rv;
      jw_kr_lr_corr += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -kr_lr_corr) / el_pair_list->get(jt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), en_lt.begin(), -kw_lr_corr) / el_pair_list->get(jt).wgt;
      jr_kr_lw_corr += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -kr_lw_corr) * el_pair_list->get(jt).rv;
      jr_kw_lw_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), ct_lt.begin(), -kw_lw_corr) * el_pair_list->get(jt).rv;
      jw_kr_lw_corr += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -kr_lw_corr) / el_pair_list->get(jt).wgt;
      jw_kw_lw_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), ct_lt.begin(), -kw_lw_corr) / el_pair_list->get(jt).wgt;
    }
    emp4       += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[0]  += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[1]  += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[2]  += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[3]  += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[4]  += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[5]  += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[6]  += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[7]  += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[8]  += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[9]  += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[10] += jw_kw_lw_corr * el_pair_list->get(it).rv;
    control[11] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
  }
}
void MP::mcmp4_energy_ik(double& emp4, std::vector<double>& control) {
  // ik contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
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

        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(jt).rv;});
        std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(jt).wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_ik_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(lt).rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(lt).wgt;});
      }

      double jr_lr_corr = 0;
      double jr_lw_corr = 0;
      double jw_lr_corr = 0;
      double jw_lw_corr = 0;
      for (auto jt = 0, lt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++, lt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_ik.h"

        jr_lr_corr += en * el_pair_list->get(jt).rv * el_pair_list->get(lt).rv;
        jr_lw_corr += en * el_pair_list->get(jt).rv / el_pair_list->get(lt).wgt;
        jw_lr_corr += en / el_pair_list->get(jt).wgt * el_pair_list->get(lt).rv;
        jw_lw_corr += en / el_pair_list->get(jt).wgt / el_pair_list->get(lt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -jr_lr_corr) * el_pair_list->get(kt).rv;
      jr_kr_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -jr_lw_corr) * el_pair_list->get(kt).rv;
      jw_kr_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_lt.begin(), -jw_lr_corr) * el_pair_list->get(kt).rv;
      jw_kr_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_lt.begin(), -jw_lw_corr) * el_pair_list->get(kt).rv;
      jr_kw_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -jr_lr_corr) / el_pair_list->get(kt).wgt;
      jr_kw_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -jr_lw_corr) / el_pair_list->get(kt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_lt.begin(), -jw_lr_corr) / el_pair_list->get(kt).wgt;
      jw_kw_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_lt.begin(), -jw_lw_corr) / el_pair_list->get(kt).wgt;
    }
    emp4        += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[12] += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[13] += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[14] += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[15] += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[16] += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[17] += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[18] += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[19] += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[20] += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[21] += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[22] += jw_kw_lw_corr * el_pair_list->get(it).rv;
    control[23] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
  }
}
void MP::mcmp4_energy_il(double& emp4, std::vector<double>& control) {
  // il contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
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

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(kt).rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(kt).wgt;});
      }

      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_il_j.h"

        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(jt).rv;});
        std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(jt).wgt;});
      }

      double jr_kr_corr = 0;
      double jr_kw_corr = 0;
      double jw_kr_corr = 0;
      double jw_kw_corr = 0;
      for (auto kt = 0, jt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, jt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_il.h"

        jr_kr_corr += en * el_pair_list->get(jt).rv  * el_pair_list->get(kt).rv;
        jr_kw_corr += en * el_pair_list->get(jt).rv  / el_pair_list->get(kt).wgt;
        jw_kr_corr += en / el_pair_list->get(jt).wgt * el_pair_list->get(kt).rv;
        jw_kw_corr += en / el_pair_list->get(jt).wgt / el_pair_list->get(kt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_kt.begin(), -jr_kr_corr) * el_pair_list->get(lt).rv;
      jr_kr_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_kt.begin(), -jr_kr_corr) / el_pair_list->get(lt).wgt;
      jr_kw_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_kt.begin(), -jr_kw_corr) * el_pair_list->get(lt).rv;
      jr_kw_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_kt.begin(), -jr_kw_corr) / el_pair_list->get(lt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_kt.begin(), -jw_kw_corr) * el_pair_list->get(lt).rv;
      jw_kw_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_kt.begin(), -jw_kw_corr) / el_pair_list->get(lt).wgt;
      jw_kr_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_kt.begin(), -jw_kr_corr) * el_pair_list->get(lt).rv;
      jw_kr_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_kt.begin(), -jw_kr_corr) / el_pair_list->get(lt).wgt;
    }
    emp4       += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[24] += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[25] += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[26] += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[27] += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[28] += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[29] += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[30] += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[31] += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[32] += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[33] += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[34] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
    control[35] += jw_kw_lw_corr * el_pair_list->get(it).rv;
  }
}
void MP::mcmp4_energy_ijkl(double& emp4, std::vector<double>& control) {
  // fourth order sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    std::array<double, 3> jr_kr_lr; jr_kr_lr.fill(0.0);
    std::array<double, 3> jr_kr_lw; jr_kr_lw.fill(0.0);
    std::array<double, 3> jr_kw_lr; jr_kw_lr.fill(0.0);
    std::array<double, 3> jr_kw_lw; jr_kw_lw.fill(0.0);
    std::array<double, 3> jw_kr_lr; jw_kr_lr.fill(0.0);
    std::array<double, 3> jw_kr_lw; jw_kr_lw.fill(0.0);
    std::array<double, 3> jw_kw_lr; jw_kw_lr.fill(0.0);
    std::array<double, 3> jw_kw_lw; jw_kw_lw.fill(0.0);
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      std::array<double, 3> kr_lr; kr_lr.fill(0.0);
      std::array<double, 3> kr_lw; kr_lw.fill(0.0);
      std::array<double, 3> kw_lr; kw_lr.fill(0.0);
      std::array<double, 3> kw_lw; kw_lw.fill(0.0);
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

          std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y){return x + y * el_pair_list->get(lt).rv;});
          std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y){return x + y / el_pair_list->get(lt).wgt;});
        }
        std::array<double, 3> en_l_t; en_l_t.fill(0.0);
        std::array<double, 3> ct_l_t; ct_l_t.fill(0.0);
#include "qc_mcmp4_ijk.h"
        for (int group = 0; group < 3; group++) {
          kr_lr[group] += en_l_t[group] * el_pair_list->get(kt).rv;
          kr_lw[group] += ct_l_t[group] * el_pair_list->get(kt).rv;
          kw_lr[group] += en_l_t[group] / el_pair_list->get(kt).wgt;
          kw_lw[group] += ct_l_t[group] / el_pair_list->get(kt).wgt;
        }
      }
      for (int group = 0; group < 3; group++) {
        jr_kr_lr[group] += kr_lr[group] * el_pair_list->get(jt).rv;
        jr_kr_lw[group] += kr_lw[group] * el_pair_list->get(jt).rv;
        jr_kw_lr[group] += kw_lr[group] * el_pair_list->get(jt).rv;
        jr_kw_lw[group] += kw_lw[group] * el_pair_list->get(jt).rv;
        jw_kr_lr[group] += kr_lr[group] / el_pair_list->get(jt).wgt;
        jw_kr_lw[group] += kr_lw[group] / el_pair_list->get(jt).wgt;
        jw_kw_lr[group] += kw_lr[group] / el_pair_list->get(jt).wgt;
        jw_kw_lw[group] += kw_lw[group] / el_pair_list->get(jt).wgt;
      }
    }
    for (int group = 0; group < 3; group++) {
      emp4             += jr_kr_lr[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 0] += jr_kr_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 1] += jw_kr_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 2] += jr_kw_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 3] += jw_kw_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 4] += jr_kr_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 5] += jr_kr_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 6] += jr_kw_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 7] += jw_kr_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 8] += jw_kr_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 9] += jr_kw_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group +10] += jw_kw_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group +11] += jw_kw_lw[group] / el_pair_list->get(it).wgt;
    }
  }
}

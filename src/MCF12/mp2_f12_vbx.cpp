#include <algorithm>
#include <iostream>
#include <numeric>

#include "cblas.h"
#include "mp2_f12_vbx.h"

MP2_F12_VBX::MP2_F12_VBX(const IOPs& iops) : MP2_F12_V(iops, "f12_VBX"),
  T_io_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_io_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_jo_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_jo_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_ko_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0)
{
  nsamp_one_3 = nsamp_one_2 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-2);
  nsamp_one_4 = nsamp_one_3 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-3);
}

void MP2_F12_VBX::energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  calculate_bx(emp, control, wavefunctions, electron_pair_list, electron_list);
}

void MP2_F12_VBX::calculate_bx(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);
  traces.update_bx(wavefunctions, electron_pair_list, electron_list);
  emp += calculate_bx_t_fa(electron_pair_list, electron_list);
  emp += calculate_bx_t_fb(electron_pair_list, electron_list);
  emp += calculate_bx_t_fc(electron_pair_list, electron_list);
  if (!correlation_factor->f12_d_is_zero()) {
    traces.update_bx_fd_traces(wavefunctions, electron_list);
    emp += calculate_bx_t_fd(electron_pair_list, electron_list);
  }
  emp += calculate_bx_k(electron_pair_list, electron_list);
}

double MP2_F12_VBX::calculate_bx_t_fa_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // This function should be nearly identical to MP2_F12_V::calculate_v_2e;
  blas_wrapper.dgekv(electron_pair_list->size(), 
      1.0, 
      correlation_factor->f12p, 1, 
      electron_pair_list->rv,  1,
      0.0, 
      T_ip_io, 1);
 
  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c1 / c3,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(electron_pair_list->size(), 
      c3, 
      traces.p11, 0, 1, 
      traces.p22, 0, 1,
      0.0, 
      T_ip_io, 2*electron_pair_list->size(), 1);
  blas_wrapper.dgekv(electron_pair_list->size(), 
      1.0, 
      T_ip_io, 2.0*electron_pair_list->size(), 1, 
      T_ip, 0, 1,
      0.0, 
      T_ip_io, electron_pair_list->size(), 1);
 
  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c2 / c4,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(electron_pair_list->size(),
      c4, 
      traces.p12, 0, 1,
      traces.p12, 0, 1,
      0.0, 
      T_ip_io, 2*electron_pair_list->size(), 1);
  blas_wrapper.dgekv(electron_pair_list->size(), 
      1.0, 
      T_ip_io, 2.0*electron_pair_list->size(), 1, 
      T_ip, 0, 1,
      1.0, 
      T_ip_io, electron_pair_list->size(), 1);
 
  double result = blas_wrapper.ddot(electron_pair_list->size(), 
      T_ip_io, 0, 1, 
      T_ip_io, electron_pair_list->size(), 1);
  result *= nsamp_pair;
  return result;
}
double MP2_F12_VBX::calculate_bx_t_fa_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // This function should be nearly identical to MP2_F12_V::calculate_v_3e;
  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      correlation_factor->f23, electron_list->size(),
      traces.k13, electron_list->size(),
      0.0, 
      T_ip_io, electron_list->size());
  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p13, electron_list->size(),
      0.0, 
      T_ip_jo, electron_list->size());
  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c1 / c3,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      traces.p22, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(), 
      c3,
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      0.0,
      T_io, 1);

  blas_wrapper.dgekm(false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0, 
      T_ip_jo, electron_list->size());
  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c2 / c4,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      traces.p12, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      c4,
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      1.0,
      T_io, 1);
  double result = blas_wrapper.ddot(electron_list->size(),
      T_io, 1,
      electron_list->inverse_weight, 1);
  result = -2.0 * result * nsamp_pair * nsamp_one_1;
  return result;
}
double MP2_F12_VBX::calculate_bx_t_fa_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // This function should be nearly identical to MP2_F12_V::calculate_v_4e;
  blas_wrapper.ddgmm(BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());
  blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      T_io_jo, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());

  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c1 / c3,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.k13, 
      traces.p23, traces.k23,
      T_ip, 
      c3, 0.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.v13, 
      traces.p23, traces.v23,
      T_ip,
      -c3, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  blas_wrapper.shift(electron_pair_list->size(),
      2.0 * c2 / c4,
      correlation_factor->f12p_a, 1,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      T_ip, 1,
      0.0, 
      T_ip, 1);
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p23, traces.k13, 
      traces.p13, traces.k23,
      T_ip, 
      c4, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo,
      traces.p23, traces.v13,
      traces.p13, traces.v23,
      T_ip, 
      -c4, 1.0,
      electron_list->size(), electron_pair_list->size());
  return blas_wrapper.accumulate(electron_list->size(), T_io, 1) * nsamp_pair * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fa(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fa_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_4e(electron_pair_list, electron_list);
  return en;
}

double MP2_F12_VBX::calculate_bx_t_fb_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double result = 0.0;

  // f12 f12b
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      correlation_factor->f12o_b, electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());

  // o11 / w1 and o22 / w2
  blas_wrapper.dgekv(
      electron_list->size(),
      1.0,
      electron_list->inverse_weight, 1,
      traces.op11, 1,
      0.0,
      T_io_ko, 1.0);

  // f12 f12b o11 / w1
  blas_wrapper.dgemv(
      true,
      electron_list->size(), electron_list->size(),
      1.0, 
      T_io_jo, electron_list->size(),
      T_io_ko, 1,
      0.0,
      T_io, 1);

  // c3 f12 f12 f12b o11 o22 / (w1 w2)
  result += c3 * blas_wrapper.ddot(electron_list->size(), T_io, 1, T_io_ko, 1);

  // f12 f12b o12
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      traces.op12, electron_list->size(),
      T_io_jo, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());
  // f12 f12b o12 o21
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      traces.op12, electron_list->size(),
      T_io_jo, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());
  // f12 f12b o12 o21 / w1 <- could be w2; don't think it matters
  blas_wrapper.dgemv(
      true,
      electron_list->size(), electron_list->size(),
      1.0, 
      T_io_jo, electron_list->size(),
      electron_list->inverse_weight, 1,
      0.0,
      T_io, 1);
  // c4 f12 f12b o12 o21 / (w1 w2) 
  result += c4 * blas_wrapper.ddot(electron_list->size(), T_io, 1, electron_list->inverse_weight, 1);
  return result * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fb_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  // f23 / w3
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_jo_ko, electron_list->size());

  // o31 k13
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      traces.op12, electron_list->size(),
      traces.ok12, electron_list->size(),
      0.0,
      T_io_ko, electron_list->size());
  // c3 f23 o31 k13 
  blas_wrapper.dgemm(
      true, false,
      electron_list->size(), electron_list->size(), electron_list->size(), 
      c3,
      T_jo_ko, electron_list->size(),
      T_io_ko, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());
  // c3 f23 o31 o22 k13 / w3
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      T_io_jo, electron_list->size(),
      traces.op11, 1,
      T_io_ko, electron_list->size());

  // f23 o32 / w3
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      T_jo_ko, electron_list->size(),
      traces.op12, electron_list->size(),
      0.0,
      T_jo_ko, electron_list->size());

  // c4 f23 o32 k31 / w3
  blas_wrapper.dgemm(
      true, false,
      electron_list->size(), electron_list->size(), electron_list->size(), 
      c4,
      T_jo_ko, electron_list->size(),
      traces.ok12, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());

  // f23 (c3 o31 o22 k13 + c4 o32 o21 k31)  / w3
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      T_io_jo, electron_list->size(),
      traces.op12, electron_list->size(),
      1.0,
      T_io_ko, electron_list->size());

  // f12b f23 (c3 o31 o22 k13 + c4 o32 o21 k31)  / w3
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      T_io_ko, electron_list->size(),
      correlation_factor->f12o_b, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());

  // f12b f23 (c3 o31 o22 k13 + c4 o32 o21 k31)  / (w2 w3)
  blas_wrapper.dgemv(
      true,
      electron_list->size(), electron_list->size(),
      1.0,
      T_io_jo, electron_list->size(),
      electron_list->inverse_weight, 1,
      0.0,
      T_io, 1);

  // f12b f23 (c3 o31 o22 k13 + c4 o32 o21 k31)  / (w1 w2 w3)
  double result = blas_wrapper.ddot(
      electron_list->size(), 
      T_io, 1,
      electron_list->inverse_weight, 1);
  return -2.0 * result * nsamp_one_3;
}
double MP2_F12_VBX::calculate_bx_t_fb_4e_help_direct(
    const std::vector<double>& S_io_ko_1, const std::vector<double>& S_io_ko_2,
    const std::vector<double>& S_jo_lo_1, const std::vector<double>& S_jo_lo_2, 
    const std::vector<double>& S_io_jo, const std::vector<double>& S_ko_lo,
    const std::vector<double>& weight,
    size_t size) {
  vector_double& R_jo_lo = T_io_ko;
  vector_double& R_jo_ko_2 = T_io_jo;

  blas_wrapper.dgekv(
      size * size,
      1.0,
      S_io_ko_1, 1,
      S_io_ko_2, 1,
      0.0,
      T_io_ko, 1);
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      size, size,
      T_io_ko, size,
      weight, 1,
      T_io_ko, size);
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      size, size,
      T_io_ko, size,
      weight, 1,
      T_io_ko, size);

  blas_wrapper.dgemm(true, false,
      size, size, size,
      1.0,
      S_ko_lo, size,
      R_jo_lo, size,
      0.0,
      T_jo_ko, size);

  blas_wrapper.dgemm(false, true,
      size, size, size,
      1.0,
      T_io_ko, size,
      S_io_jo, size,
      0.0,
      R_jo_ko_2, size);

  blas_wrapper.dgekv(
      size * size,
      1.0,
      T_jo_ko, 1,
      R_jo_ko_2, 1,
      0.0,
      T_jo_ko, 1);
  blas_wrapper.dgekm(
      false, true,
      size, size, 
      1.0,
      R_jo_lo, size,
      S_io_jo, size,
      0.0,
      R_jo_ko_2, size);
  blas_wrapper.dgekv(
      size * size, 
      1.0, 
      T_io_ko, 1,
      S_ko_lo, 1,
      0.0,
      T_io_ko, 1);


  blas_wrapper.dgemm(false, false,
      size, size, size,
      -1.0,
      T_io_ko, size,
      R_jo_ko_2, size,
      1.0,
      T_jo_ko, size);

  blas_wrapper.dscal(
      size,
      0.0,
      T_jo_ko, size + 1);
  return blas_wrapper.accumulate(T_jo_ko.size(), T_jo_ko, 1);
}
double MP2_F12_VBX::calculate_bx_t_fb_4e_help_exchange(
    const std::vector<double>& S_io_jo, const std::vector<double>& S_io_ko,
    const std::vector<double>& S_io_lo, const std::vector<double>& S_jo_ko,
    const std::vector<double>& S_jo_lo, const std::vector<double>& S_ko_lo,
    const std::vector<double>& weight,
    size_t size) {
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      size, size,
      S_ko_lo, size,
      weight, 1,
      T_ko_lo, size);
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      size, size,
      S_io_ko, size,
      weight, 1,
      T_io_ko, size);

  double result = 0.0;
  for (int io = 0; io < size; io++) {
    blas_wrapper.ddgmm(
        BLAS_WRAPPER::LEFT_SIDE,
        size, size,
        S_jo_lo, 0,  size,
        S_io_lo, io * size, 1,
        T_jo_lo, 0, size);
    blas_wrapper.dgemm(
        true, false, 
        size, size, size,
        1.0,
        T_ko_lo, size,
        T_jo_lo, size,
        0.0,
        T_jo_ko, size);
    blas_wrapper.dgekm(
        false, false,
        size, size,
        1.0,
        T_jo_ko, size,
        S_jo_ko, size,
        0.0,
        T_jo_ko, size);
    blas_wrapper.dgemv(
        true,
        size, size,
        1.0,
        T_jo_ko, 0, size,
        T_io_ko, io * size , 1,
        0.0,
        T_io_jo, io * size, 1);
  }
  blas_wrapper.dgekm(
      false, false,
      size, size,
      1.0,
      T_io_jo, size,
      S_io_jo, size,
      0.0,
      T_io_jo, size);
  blas_wrapper.dgemv(
      true,
      size, size,
      1.0, 
      T_io_jo, size, 
      weight, 1,
      0.0,
      T_io, 1);
  return blas_wrapper.ddot(size, T_io, 1, weight, 1);
}
double MP2_F12_VBX::calculate_bx_t_fb_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  // f34 f12b o31 o42 k13 k24 or (inorder of the arugment) k13 o21 k24 o42 f12b f23
  t[0] += calculate_bx_t_fb_4e_help_direct(traces.ok12, traces.op12, traces.ok12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);
  // f34 f12b o31 o42 v13 v24 or (inorder of the arugment) v13 o21 v24 o42 f12b f23
  t[0] -= calculate_bx_t_fb_4e_help_direct(traces.ov12, traces.op12, traces.ov12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);

  // f34 f12b o32 o41 k13 k24 or (inorder of the arugment) f12b k13 o41 o32 k24 f34
  t[1] += calculate_bx_t_fb_4e_help_exchange(correlation_factor->f12o_b, traces.ok12, traces.op12, traces.op12, traces.ok12, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);
  // f34 f12b o32 o41 v13 v24 or (inorder of the arugment) f12b v13 o41 o32 v24 f34
  t[1] -= calculate_bx_t_fb_4e_help_exchange(correlation_factor->f12o_b, traces.ov12, traces.op12, traces.op12, traces.ov12, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_4;
}
double MP2_F12_VBX::calculate_bx_t_fb(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fb_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fb_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fb_4e(electron_pair_list, electron_list);
  return en;
}

double MP2_F12_VBX::calculate_bx_t_fc_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      c3, 
      traces.p11, 1,
      traces.dp22, 1,
      0.0,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      -c3, 
      traces.dp11, 1,
      traces.p22, 1,
      1.0,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      c4, 
      traces.dp12, 1,
      traces.p12, 1,
      1.0,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      -c4, 
      traces.p12, 1,
      traces.dp21, 1,
      1.0,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0, 
      T_ip, 1,
      correlation_factor->f12p, 1,
      0.0,
      T_ip, 1);
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0, 
      T_ip, 1,
      correlation_factor->f12p_c, 1,
      0.0,
      T_ip, 1);
  return nsamp_pair * blas_wrapper.ddot(
      electron_pair_list->size(),
      T_ip, 1,
      electron_pair_list->rv, 1);
}
double MP2_F12_VBX::calculate_bx_t_fc_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      correlation_factor->f12p_c, 1,
      electron_pair_list->rv, 1,
      0.0,
      T_ip, 1);
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      correlation_factor->f23, electron_list->size(),
      traces.k13, electron_list->size(),
      0.0,
      T_ip_io, electron_list->size());
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_pair_list->size(),
      T_ip_io, electron_list->size(),
      T_ip, 1,
      T_ip_io, electron_list->size());

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p13, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      c3,
      T_ip_jo, electron_list->size(), 
      traces.dp22, 1,
      0.0,
      T_io, 1);

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.dp31, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      -c3,
      T_ip_jo, electron_list->size(), 
      traces.p22, 1,
      1.0,
      T_io, 1);

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.dp32, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      c4,
      T_ip_jo, electron_list->size(), 
      traces.p12, 1,
      1.0,
      T_io, 1);

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(),
      -c4,
      T_ip_jo, electron_list->size(), 
      traces.dp21, 1,
      1.0,
      T_io, 1);

  return -2.0 * blas_wrapper.ddot(electron_list->size(), T_io, 1, electron_list->inverse_weight, 1) * nsamp_pair * nsamp_one_1;
}
double MP2_F12_VBX::calculate_bx_t_fc_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_list->size(),
      T_io_jo, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_jo, electron_list->size());
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      correlation_factor->f12p_c, 1,
      electron_pair_list->rv, 1,
      0.0,
      T_ip, 1);

  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.k13, 
      traces.dp32, traces.k23,
      T_ip,
      c3, 0.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.dp31, traces.k13, 
      traces.p23, traces.k23,
      T_ip, 
      -c3, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p13, traces.v13, 
      traces.dp32, traces.v23,
      T_ip, 
      -c3, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.dp31, traces.v13, 
      traces.p23, traces.v23,
      T_ip,
      c3, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.dp32, traces.k13, 
      traces.p13, traces.k23,
      T_ip,
      c4, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p23, traces.k13, 
      traces.dp31, traces.k23,
      T_ip, 
      -c4, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.dp32, traces.v13, 
      traces.p13, traces.v23,
      T_ip, 
      -c4, 1.0,
      electron_list->size(), electron_pair_list->size());
 
  calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, 
      traces.p23, traces.v13, 
      traces.dp31, traces.v23,
      T_ip,
      c4, 1.0,
      electron_list->size(), electron_pair_list->size());
  return blas_wrapper.accumulate(electron_list->size(), T_io, 1) * nsamp_pair * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fc(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fc_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fc_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fc_4e(electron_pair_list, electron_list);
  return en;
}

double MP2_F12_VBX::calculate_bx_t_fd_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
      t_io[0] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op11[io] * traces.ds_p22[io * traces.electrons + jo] - traces.ds_p11[io * traces.electrons + jo] * traces.op11[jo]) * electron_list->inverse_weight[jo];
      t_io[1] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op12[io * traces.electrons + jo] * traces.ds_p12[io * traces.electrons + jo] - traces.ds_p21[io * traces.electrons + jo] * traces.op12[io * traces.electrons + jo]) * electron_list->inverse_weight[jo];
    }
    t[0] += t_io[0] * electron_list->inverse_weight[io];
    t[1] += t_io[1] * electron_list->inverse_weight[io];
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fd_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            t_jo[0] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.op12[io * traces.electrons + ko]       * electron_list->inverse_weight[ko];
            t_jo[1] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.ds_p31[(io * traces.electrons + jo) * traces.electrons + ko] * electron_list->inverse_weight[ko];
            t_jo[2] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.ds_p32[(io * traces.electrons + jo) * traces.electrons + ko] * electron_list->inverse_weight[ko];
            t_jo[3] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.op12[jo * traces.electrons + ko]       * electron_list->inverse_weight[ko];
          }
        }
        auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
        t_io[0] += (t_jo[0] * traces.ds_p22[io * traces.electrons + jo]   - t_jo[1] * traces.op11[jo]      ) * f_d * electron_list->inverse_weight[jo];
        t_io[1] += (t_jo[2] * traces.op12[io * traces.electrons + jo]     - t_jo[3] * traces.ds_p21[io * traces.electrons + jo]) * f_d * electron_list->inverse_weight[jo];
      }
    }
    t[0] += t_io[0] * electron_list->inverse_weight[io];
    t[1] += t_io[1] * electron_list->inverse_weight[io];
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_one_3;
}
double MP2_F12_VBX::calculate_bx_t_fd_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0, 0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        if (ko != jo) {
          std::array<double, 8> t_ko{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
          for (int lo = 0; lo < electron_list->size(); ++lo) {
            if (lo != io) {
              auto wgt = electron_list->inverse_weight[lo];
              t_ko[0] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p32[(io * traces.electrons + jo) * traces.electrons + lo]                     * traces.ok12[jo * traces.electrons + lo] * wgt;
              t_ko[1] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[jo * traces.electrons + lo]       * traces.ok12[jo * traces.electrons + lo] * wgt;

              t_ko[2] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p32[(io * traces.electrons + jo) * traces.electrons + lo]                     * traces.ov12[jo * traces.electrons + lo] * wgt;
              t_ko[3] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[jo * traces.electrons + lo]       * traces.ov12[jo * traces.electrons + lo] * wgt;

              t_ko[4] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[io * traces.electrons + lo]       * traces.ok12[jo * traces.electrons + lo] * wgt;
              t_ko[5] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p31[(io * traces.electrons + jo) * traces.electrons + lo]                     * traces.ok12[jo * traces.electrons + lo] * wgt;

              t_ko[6] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[io * traces.electrons + lo]       * traces.ov12[jo * traces.electrons + lo] * wgt;
              t_ko[7] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p31[(io * traces.electrons + jo) * traces.electrons + lo]                     * traces.ov12[jo * traces.electrons + lo] * wgt;
            }
          }
          t_jo[0] += (t_ko[0] * traces.op12[io * traces.electrons + ko] - t_ko[1] * traces.ds_p31[(io * traces.electrons + jo) * traces.electrons + ko]) * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[1] += (t_ko[2] * traces.op12[io * traces.electrons + ko] - t_ko[3] * traces.ds_p31[(io * traces.electrons + jo) * traces.electrons + ko]) * traces.ov12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[2] += (t_ko[4] * traces.ds_p32[(io * traces.electrons + jo) * traces.electrons + ko]              - t_ko[5] * traces.op12[jo * traces.electrons + ko]) * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[3] += (t_ko[6] * traces.ds_p32[(io * traces.electrons + jo) * traces.electrons + ko]              - t_ko[7] * traces.op12[jo * traces.electrons + ko]) * traces.ov12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
        }
      }
      auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
      t_io[0] += t_jo[0] * f_d * electron_list->inverse_weight[jo];
      t_io[1] += t_jo[1] * f_d * electron_list->inverse_weight[jo];
      t_io[2] += t_jo[2] * f_d * electron_list->inverse_weight[jo];
      t_io[3] += t_jo[3] * f_d * electron_list->inverse_weight[jo];
    }
    t[0] += t_io[0] * electron_list->inverse_weight[io];
    t[0] -= t_io[1] * electron_list->inverse_weight[io];
    t[1] += t_io[2] * electron_list->inverse_weight[io];
    t[1] -= t_io[3] * electron_list->inverse_weight[io];
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_4;
}
double MP2_F12_VBX::calculate_bx_t_fd(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fd_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fd_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fd_4e(electron_pair_list, electron_list);
  return en;
}

double MP2_F12_VBX::calculate_bx_k_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  blas_wrapper.dgeam(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      correlation_factor->f23, electron_list->size(),
      -1.0,
      correlation_factor->f13, electron_list->size(),
      T_ip_io, electron_list->size());

  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      traces.k12, 1,
      electron_pair_list->rv, 1,
      0.0,
      T_ip, 1);

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      T_ip_io, electron_list->size(),
      correlation_factor->f23, electron_list->size(),
      0.0,
      T_ip_io, electron_list->size());

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      T_ip_io, electron_list->size(),
      traces.p13, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      T_ip_jo, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());

  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(), 
      2.0 * c4 * nsamp_pair * nsamp_one_1,
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      0.0,
      T_io, 1);

  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      T_ip, 1,
      traces.p12, 1,
      0.0,
      T_ip, 1);

  blas_wrapper.dgemv(
      false,
      electron_list->size(), electron_pair_list->size(), 
      2.0 * c3 * nsamp_pair * nsamp_one_1,
      T_ip_io, electron_list->size(),
      T_ip, 1,
      0.0,
      T_io_jo, 1);

  blas_wrapper.dgekv(
      electron_list->size(),
      1.0,
      T_io_jo, 1,
      traces.op11, 1,
      1.0,
      T_io, 1);

  std::array<double, 2> t{0.0, 0.0};
  return blas_wrapper.ddot(electron_list->size(), T_io, 1, electron_list->inverse_weight, 1);
}
void MP2_F12_VBX::calculate_bx_k_4e_help(
    size_t electrons, size_t electron_pairs, double alpha,
    const std::vector<double>& S_ip_io, const std::vector<double>& S_ip_jo,
    const std::vector<double>& S_io_jo, std::vector<double>& T_io_jo,
    const std::vector<double>& S_jo, std::vector<double>& T_io) {
  blas_wrapper.dgemm(
      false, true,
      electrons, electrons, electron_pairs,
      1.0,
      S_ip_jo, electrons,
      S_ip_io, electrons,
      0.0, 
      T_io_jo, electrons);
  blas_wrapper.dgekm(
      false, false,
      electrons, electrons,
      1.0,
      T_io_jo, electrons,
      S_io_jo, electrons,
      0.0,
      T_io_jo, electrons);
  blas_wrapper.dgemv(
      true,
      electrons, electrons,
      alpha,
      T_io_jo, electrons,
      S_jo, 1,
      1.0, 
      T_io, 1);

}
double MP2_F12_VBX::calculate_bx_k_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  blas_wrapper.vfill(electron_list->size(), 0.0, T_io, 1);
  // T_ip = k12 / (r12 w12)
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      traces.k12, 1,
      0.0,
      T_ip, 1);
  // T_ip_jo = (f14 - f24)
  blas_wrapper.dgeam(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      correlation_factor->f13, electron_list->size(),
      -1.0,
      correlation_factor->f23, electron_list->size(),
      T_ip_jo, electron_list->size());
  // T_ip_jo = (f14 - f24) k12 / (r12 w12)
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_pair_list->size(),
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      T_ip_jo, electron_list->size());
  // T_ip_io = o23 k13
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      traces.p23, electron_list->size(),
      traces.k13, electron_list->size(),
      0.0,
      T_ip_io, electron_list->size());
  // T_io_lo = o44 / (w4)
  blas_wrapper.dgekv(
      electron_list->size(),
      1.0,
      electron_list->inverse_weight, 1,
      traces.op11, 1,
      0.0,
      T_io_lo, 1);
  // (f14 - f24) o44 k12 o23 k13 f34
  calculate_bx_k_4e_help(
      electron_list->size(), electron_pair_list->size(), c3,
      T_ip_io, T_ip_jo,
      correlation_factor->f12o, T_jo_lo,
      T_io_lo, T_io);

  // T_ip_jo = (f14 - f24) k12 p24 / (r12 w12)
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_jo, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  // T_io_jo = f34 p34
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      correlation_factor->f12o, electron_list->size(),
      traces.op12, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());
  calculate_bx_k_4e_help(
      electron_list->size(), electron_pair_list->size(), c4,
      traces.k13, T_ip_jo,
      T_io_jo, T_jo_lo,
      electron_list->inverse_weight, T_io);
  
  // T_ip_jo = (f23 - f13)
  blas_wrapper.dgeam(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      correlation_factor->f23, electron_list->size(),
      -1.0,
      correlation_factor->f13, electron_list->size(),
      T_ip_io, electron_list->size());
  // T_ip_io = (f23 - f13) k12 / (r12 w12)
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_pair_list->size(),
      T_ip_io, electron_list->size(),
      T_ip, 1,
      T_ip_io, electron_list->size());
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_list->size(),
      1.0,
      traces.ok12, electron_list->size(),
      traces.op12, electron_list->size(),
      0.0,
      T_io_jo, electron_list->size());
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_pair_list->size(),
      correlation_factor->f23, electron_list->size(),
      traces.p12, 1,
      T_ip_jo, electron_list->size());
  calculate_bx_k_4e_help(
      electron_list->size(), electron_pair_list->size(), c3,
      T_ip_io, T_ip_jo,
      T_io_jo, T_jo_lo,
      electron_list->inverse_weight, T_io);

  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      T_ip_io, electron_list->size(),
      traces.p23, electron_list->size(),
      0.0,
      T_ip_io, electron_list->size());
  blas_wrapper.dgekm(
      false, false,
      electron_list->size(), electron_pair_list->size(),
      1.0,
      correlation_factor->f23, electron_list->size(),
      traces.p13, electron_list->size(),
      0.0,
      T_ip_jo, electron_list->size());
  calculate_bx_k_4e_help(
      electron_list->size(), electron_pair_list->size(), c4,
      T_ip_io, T_ip_jo,
      traces.ok12, T_jo_lo,
      electron_list->inverse_weight, T_io);

  double result = blas_wrapper.ddot(electron_list->size(), T_io, 1, electron_list->inverse_weight, 1);
  return -2.0 * result * nsamp_pair * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_k_5e_direct_help(
    double alpha,
    const std::vector<double>& S_ip_io_1, const std::vector<double>& S_ip_io_2,
    const std::vector<double>& S_jo_ko_1, const std::vector<double>& S_jo_ko_2,
    const std::vector<double>& S_io_ko, const std::vector<double>& S_ip_jo,
    const std::vector<double>& weight,
    size_t size, size_t size_ep) {
  blas_wrapper.dgekm(
      false, false,
      size, size_ep,
      1.0,
      S_ip_io_1, size,
      S_ip_io_2, size,
      0.0,
      T_ip_io, size);
  blas_wrapper.dgekm(
      false, false,
      size, size,
      1.0,
      S_jo_ko_1, size,
      S_jo_ko_2, size,
      0.0,
      T_jo_ko, size);

  blas_wrapper.dgemm(false, true,
      size, size, size_ep,
      1.0,
      S_ip_jo, size,
      T_ip_io, size,
      0.0,
      T_io_jo, size);

  blas_wrapper.dscal(size, 0.0, T_io_jo, size+1);

  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      size, size, 
      T_jo_ko, size, 
      weight, 1,
      T_jo_ko, size);

  blas_wrapper.dgemm(true, false,
      size, size, size,
      1.0,
      T_jo_ko, size,
      S_io_ko, size,
      0.0,
      T_io_ko, size);

  blas_wrapper.dgekm(
      false, false,
      size, size,
      1.0,
      T_io_jo, size,
      T_io_ko, size,
      0.0,
      T_io_jo, size);

  blas_wrapper.dgemv(
      true,
      size, size, 
      alpha,
      T_io_jo, size, 
      weight, 1,
      0.0,
      T_io, 1);

  double en = blas_wrapper.ddot(size, T_io, 1, weight, 1);
  return en;
}
double MP2_F12_VBX::calculate_bx_k_5e_exchange_help(
    double alpha,
    const vector_double& S_ip_io,
    const vector_double& S_ip_jo,
    const vector_double& S_ip_ko,
    const vector_double& S_io_jo,
    const vector_double& S_io_ko,
    const vector_double& S_jo_ko,
    const vector_double& weight,
    size_t electrons, size_t electron_pairs
    ) {
  for (int io = 0; io < electrons; io++) {
    blas_wrapper.ddgmm(
        BLAS_WRAPPER::RIGHT_SIDE,
        electrons, electron_pairs,
        S_ip_ko, 0, electrons,
        S_ip_io, io, electrons,
        T_ip_io, 0, electrons);
    blas_wrapper.dgemm(
        false, true,
        electrons, electrons, electron_pairs,
        1.0,
        T_ip_io, electrons,
        S_ip_jo, electrons,
        0.0, 
        T_jo_ko, electrons);
    blas_wrapper.dgekm(
        false, false,
        electrons, electrons,
        1.0,
        T_jo_ko, electrons,
        S_jo_ko, electrons,
        0.0,
        T_jo_ko, electrons);
    blas_wrapper.dgemv(
        true,
        electrons, electrons,
        1.0,
        T_jo_ko, 0, electrons,
        S_io_ko, electrons * io, 1,
        0.0,
        T_io_jo, electrons * io, 1);
  }
  blas_wrapper.dgekm(
      false, false,
      electrons, electrons,
      1.0,
      T_io_jo, electrons,
      S_io_jo, electrons,
      0.0,
      T_io_jo, electrons);
  blas_wrapper.dgemv(
      true,
      electrons, electrons,
      alpha,
      T_io_jo, electrons,
      weight, 1,
      0.0,
      T_io, 1);
  return blas_wrapper.ddot(electrons, T_io, 1, weight, 1);
}
double MP2_F12_VBX::calculate_bx_k_5e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double t;
  blas_wrapper.dgekv(
      electron_pair_list->size(),
      1.0,
      electron_pair_list->rv, 1,
      traces.k12, 1,
      0.0,
      T_ip, 1);
  blas_wrapper.dgeam(
      false, false,
      electron_list->size(), electron_pair_list->size(), 
      1.0,
      correlation_factor->f13, electron_list->size(),
      -1.0,
      correlation_factor->f23, electron_list->size(),
      T_ip_jo, electron_list->size());
  blas_wrapper.ddgmm(
      BLAS_WRAPPER::RIGHT_SIDE,
      electron_list->size(), electron_pair_list->size(),
      T_ip_jo, electron_list->size(),
      T_ip, 1,
      T_ip_jo, electron_list->size());

  t += calculate_bx_k_5e_direct_help(c3, traces.k13, traces.p23, traces.ok12, traces.op12, correlation_factor->f12o, T_ip_jo, 
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());
  t += calculate_bx_k_5e_direct_help(-c3, traces.v13, traces.p23, traces.ov12, traces.op12, correlation_factor->f12o, T_ip_jo, 
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());

  blas_wrapper.ddgmm(
      BLAS_WRAPPER::LEFT_SIDE,
      electron_list->size(), electron_list->size(),
      correlation_factor->f12o, electron_list->size(),
      electron_list->inverse_weight, 1,
      T_io_ko, electron_list->size());

  t += calculate_bx_k_5e_exchange_help(c4, traces.k13, T_ip_jo, traces.p23, traces.op12, T_io_ko, traces.ok12,
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());

  t += calculate_bx_k_5e_exchange_help(-c4, traces.v13, T_ip_jo, traces.p23, traces.op12, T_io_ko, traces.ov12,
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());

  return 2.0 * (t) * nsamp_pair * nsamp_one_3;
}
double MP2_F12_VBX::calculate_bx_k(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_k_3e(electron_pair_list, electron_list);
  en += calculate_bx_k_4e(electron_pair_list, electron_list);
  en += calculate_bx_k_5e(electron_pair_list, electron_list);
  return en;
}

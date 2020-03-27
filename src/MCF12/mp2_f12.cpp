#include <algorithm>
#include <iostream>
#include <numeric>

#include "cblas.h"
#include "mp2_f12.h"

MP2_F12_V::MP2_F12_V(const IOPs& iops, const Basis& basis, std::string extension) :
    MCMP(0, 0, extension, true),
    traces(basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2, iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS])
{
  correlation_factor = create_correlation_factor(iops);
  nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
  nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
}

MP2_F12_V::~MP2_F12_V() {
  delete correlation_factor;
}

void MP2_F12_V::energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_v(emp, control, wavefunctions, electron_pair_list, electron_list);
}

std::array<double, 2> MP2_F12_V::calculate_v_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> v_1_pair_0_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    auto f_12 = correlation_factor->f12p[ip];
    v_1_pair_0_one_ints[0] = v_1_pair_0_one_ints[0] + f_12 * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    v_1_pair_0_one_ints[1] = v_1_pair_0_one_ints[1] + f_12 * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_0_one_int : v_1_pair_0_one_ints) {
    v_1_pair_0_one_int *= nsamp_pair;
  }
  return v_1_pair_0_one_ints;
}
std::array<double, 2> MP2_F12_V::calculate_v_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> v_1_pair_1_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] = t[0] + correlation_factor->f23[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] = t[1] + correlation_factor->f23[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    v_1_pair_1_one_ints[0] = v_1_pair_1_one_ints[0] + t[0] * traces.p22[ip] * electron_pair_list->rv[ip];
    v_1_pair_1_one_ints[1] = v_1_pair_1_one_ints[1] + t[1] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_1_one_int : v_1_pair_1_one_ints) {
    v_1_pair_1_one_int *= nsamp_pair * nsamp_one_1;
  }
  return v_1_pair_1_one_ints;
}
std::array<double, 2> MP2_F12_V::calculate_v_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> v_1_pair_2_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        s[0] = s[0] + correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[1] = s[1] + correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[2] = s[2] + correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[3] = s[3] + correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
      }
      t[0] = t[0] + traces.p13[ip * traces.electrons + io] * (s[0] * traces.k13[ip * traces.electrons + io] - s[2] * traces.v13[ip * traces.electrons + io]) * electron_list->inverse_weight[io];
      t[1] = t[1] + traces.p23[ip * traces.electrons + io] * (s[1] * traces.k13[ip * traces.electrons + io] - s[3] * traces.v13[ip * traces.electrons + io]) * electron_list->inverse_weight[io];
    }
    v_1_pair_2_one_ints[0] = v_1_pair_2_one_ints[0] + t[0] * electron_pair_list->rv[ip];
    v_1_pair_2_one_ints[1] = v_1_pair_2_one_ints[1] + t[1] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_2_one_int : v_1_pair_2_one_ints) {
    v_1_pair_2_one_int *= nsamp_pair * nsamp_one_2;
  }
  return v_1_pair_2_one_ints;
}
void MP2_F12_V::calculate_v(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);
  auto v_1_pair_0_one_ints = calculate_v_2e(electron_pair_list, electron_list);
  auto v_1_pair_1_one_ints = calculate_v_3e(electron_pair_list, electron_list);
  auto v_1_pair_2_one_ints = calculate_v_4e(electron_pair_list, electron_list);
  emp +=  c1 * (v_1_pair_0_one_ints[0] + v_1_pair_2_one_ints[0] - 2 * v_1_pair_1_one_ints[0])
        + c2 * (v_1_pair_0_one_ints[1] + v_1_pair_2_one_ints[1] - 2 * v_1_pair_1_one_ints[1]);
}

MP2_F12_VBX::MP2_F12_VBX(const IOPs& iops, const Basis& basis) : MP2_F12_V(iops, basis, "f12_VBX"),
  T_ip(iops.iopns[KEYS::ELECTRON_PAIRS], 0.0),
  T_ip_io(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_ip_jo(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_jo_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_io_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_io_jo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0)
{
  nsamp_one_3 = nsamp_one_2 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-2);
  nsamp_one_4 = nsamp_one_3 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-3);
}

void MP2_F12_VBX::energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double e_v = 0;
  std::vector<double> c_v, c_bx;
  calculate_v(e_v, c_v, wavefunctions, electron_pair_list, electron_list);
  calculate_bx(emp, c_bx, wavefunctions, electron_pair_list, electron_list);
  emp += 2 * e_v;
}

void MP2_F12_VBX::calculate_bx(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  traces.update_bx(wavefunctions, electron_pair_list, electron_list);
  zero();
  calculate_bx_t_fa(electron_pair_list, electron_list);
  calculate_bx_t_fb(electron_pair_list, electron_list);
  calculate_bx_t_fc(electron_pair_list, electron_list);
  if (!correlation_factor->f12_d_is_zero()) {
    traces.update_bx_fd_traces(wavefunctions, electron_list);
    calculate_bx_t_fd(electron_pair_list, electron_list);
  }
  

  calculate_bx_k(electron_pair_list, electron_list);
  normalize();

  auto direct_contribution = direct_1_pair_0_one_ints[0] + direct_1_pair_0_one_ints[1]
      + direct_0_pair_2_one_ints[0] + direct_0_pair_2_one_ints[1]
      - 2.0*direct_1_pair_1_one_ints[0] - 2.0*direct_1_pair_1_one_ints[1] + 2.0*direct_1_pair_1_one_ints[2]
      - 2.0*direct_0_pair_3_one_ints[0] -2.0*direct_0_pair_3_one_ints[1]
      + direct_1_pair_2_one_ints[0] + direct_1_pair_2_one_ints[1] - direct_1_pair_2_one_ints[2] - direct_1_pair_2_one_ints[3] - 2.0*direct_1_pair_2_one_ints[4] - 2.0*direct_1_pair_2_one_ints[5]
      + direct_0_pair_4_one_ints[0] - direct_0_pair_4_one_ints[1] + direct_0_pair_4_one_ints[2] - direct_0_pair_4_one_ints[3]
      + 2.0*direct_1_pair_3_one_ints[0];
  auto xchang_contrubtion = xchang_1_pair_0_one_ints[0] + xchang_1_pair_0_one_ints[1]
    + xchang_0_pair_2_one_ints[0] + xchang_0_pair_2_one_ints[1]
      - 2.0*xchang_1_pair_1_one_ints[0] - 2.0*xchang_1_pair_1_one_ints[1] + 2.0*xchang_1_pair_1_one_ints[2]
      - 2.0*xchang_0_pair_3_one_ints[0] - 2.0*xchang_0_pair_3_one_ints[1]
      + xchang_1_pair_2_one_ints[0] + xchang_1_pair_2_one_ints[1] - xchang_1_pair_2_one_ints[2] - xchang_1_pair_2_one_ints[3] - 2.0*xchang_1_pair_2_one_ints[4] - 2.0*xchang_1_pair_2_one_ints[5]
      + xchang_0_pair_4_one_ints[0] - xchang_0_pair_4_one_ints[1] + xchang_0_pair_4_one_ints[2] - xchang_0_pair_4_one_ints[3]
      + 2.0*xchang_1_pair_3_one_ints[0];
  emp += c3 * direct_contribution + c4 * xchang_contrubtion;
}

void MP2_F12_VBX::zero() {
  direct_1_pair_0_one_ints.fill(0.0);
  direct_0_pair_2_one_ints.fill(0.0);
  direct_1_pair_1_one_ints.fill(0.0);
  direct_0_pair_3_one_ints.fill(0.0);
  direct_1_pair_2_one_ints.fill(0.0);
  direct_0_pair_4_one_ints.fill(0.0);
  direct_1_pair_3_one_ints.fill(0.0);

  xchang_1_pair_0_one_ints.fill(0.0);
  xchang_0_pair_2_one_ints.fill(0.0);
  xchang_1_pair_1_one_ints.fill(0.0);
  xchang_0_pair_3_one_ints.fill(0.0);
  xchang_1_pair_2_one_ints.fill(0.0);
  xchang_0_pair_4_one_ints.fill(0.0);
  xchang_1_pair_3_one_ints.fill(0.0);
}

void MP2_F12_VBX::normalize() {
  for (auto &it : direct_1_pair_0_one_ints) {
    it *= nsamp_pair;
  }
  for (auto &it : xchang_1_pair_0_one_ints) {
    it *= nsamp_pair;
  }
  for (auto &it : direct_0_pair_2_one_ints) {
    it *= nsamp_one_2;
  }
  for (auto &it : xchang_0_pair_2_one_ints) {
    it *= nsamp_one_2;
  }
  for (auto &it : direct_1_pair_1_one_ints) {
    it *= nsamp_pair * nsamp_one_1;
  }
  for (auto &it : xchang_1_pair_1_one_ints) {
    it *= nsamp_pair * nsamp_one_1;
  }
  for (auto &it : direct_0_pair_3_one_ints) {
    it *= nsamp_one_3;
  }
  for (auto &it : xchang_0_pair_3_one_ints) {
    it *= nsamp_one_3;
  }
  for (auto &it : direct_1_pair_2_one_ints) {
    it *= nsamp_pair * nsamp_one_2;
  }
  for (auto &it : xchang_1_pair_2_one_ints) {
    it *= nsamp_pair * nsamp_one_2;
  }
  for (auto &it : direct_0_pair_4_one_ints) {
    it *= nsamp_one_4;
  }
  for (auto &it : xchang_0_pair_4_one_ints) {
    it *= nsamp_one_4;
  }
  for (auto &it : direct_1_pair_3_one_ints) {
    it *= nsamp_pair * nsamp_one_3;
  }
  for (auto &it : xchang_1_pair_3_one_ints) {
    it *= nsamp_pair * nsamp_one_3;
  }
}

void MP2_F12_VBX::calculate_bx_t_fa_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    auto f_a = correlation_factor->f12p_a[ip];
    auto f_12 = correlation_factor->f12p[ip];
    direct_1_pair_0_one_ints[0] += f_12 * f_a * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_0_one_ints[0] += f_12 * f_a * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fa_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_a = correlation_factor->f12p_a[ip];
    direct_1_pair_1_one_ints[0] += t[0] * f_a * traces.p22[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[0] += t[1] * f_a * traces.p12[ip] * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fa_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for(int jo = 0; jo < electron_list->size();++jo) {
        s[0] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[1] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[2] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[3] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
      }
      t[0] += s[0] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] += s[1] * traces.p13[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[2] += s[2] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[3] += s[3] * traces.p23[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_a = correlation_factor->f12p_a[ip];
    direct_1_pair_2_one_ints[0] += t[0] * f_a * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[2] += t[1] * f_a * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[0] += t[2] * f_a * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[2] += t[3] * f_a * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fa(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_bx_t_fa_2e(electron_pair_list, electron_list);
  calculate_bx_t_fa_3e(electron_pair_list, electron_list);
  calculate_bx_t_fa_4e(electron_pair_list, electron_list);
}

void MP2_F12_VBX::calculate_bx_t_fb_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      auto f_b = correlation_factor->f12o_b[io * traces.electrons + jo];
      t_io[0] += f_b * correlation_factor->f12o[io * traces.electrons + jo] * traces.op11[jo] * electron_list->inverse_weight[jo];
      t_io[1] += f_b * correlation_factor->f12o[io * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
    }
    direct_0_pair_2_one_ints[0] += t_io[0] * traces.op11[io] * electron_list->inverse_weight[io];
    xchang_0_pair_2_one_ints[0] += t_io[1] * electron_list->inverse_weight[io];
  }
}
void MP2_F12_VBX::calculate_bx_t_fb_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      std::array<double, 2> t_jo{0.0, 0.0};
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        t_jo[0] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.op12[io * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
        t_jo[1] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.op12[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
      }
      auto f_b = correlation_factor->f12o_b[io * traces.electrons + jo];
      t_io[0] += t_jo[0] * f_b * traces.op11[jo] * electron_list->inverse_weight[jo];
      t_io[1] += t_jo[1] * f_b * traces.op12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
    }
    direct_0_pair_3_one_ints[0] += t_io[0] * electron_list->inverse_weight[io];
    xchang_0_pair_3_one_ints[0] += t_io[1] * electron_list->inverse_weight[io];
  }
}
double MP2_F12_VBX::calculate_bx_t_fb_4e_help(
    const std::vector<double>& S_io_ko_1, const std::vector<double>& S_io_ko_2,
    const std::vector<double>& S_io_jo, const std::vector<double>& S_ko_lo,
    const std::vector<double>& weight,
    size_t size) {
  std::vector<double>& T_jo_lo = T_io_ko;
  std::vector<double>& T_jo_ko_2 = T_io_jo;
  std::transform(S_io_ko_1.begin(), S_io_ko_1.end(), S_io_ko_2.begin(), T_io_ko.begin(), std::multiplies<>());
  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      T_io_ko[io * size + ko] *= weight[ko] * weight[io];
    }
  }

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      size, size, size,
      1.0,
      S_ko_lo.data(), size,
      T_jo_lo.data(), size,
      0.0,
      T_jo_ko.data(), size);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      size, size, size,
      1.0,
      T_io_ko.data(), size,
      S_io_jo.data(), size,
      0.0,
      T_jo_ko_2.data(), size);

  std::transform(T_jo_ko.begin(), T_jo_ko.end(), T_jo_ko_2.begin(), T_jo_ko.begin(), std::multiplies<>());
  for (int jo = 0; jo < size; ++jo) {
    for (int io = 0; io < size; ++io) {
      T_jo_ko_2[jo * size + io] = T_jo_lo[jo * size + io] * S_io_jo[io * size + jo];
    }
  }
  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      T_io_ko[io * size + ko] *= S_ko_lo[ko * size + io];
    }
  }


  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      size, size, size,
      -1.0,
      T_io_ko.data(), size,
      T_jo_ko_2.data(), size,
      1.0,
      T_jo_ko.data(), size);

  for (int jo = 0; jo < size; ++jo) {
    T_jo_ko[jo * size + jo] = 0.0;
  }
  return std::accumulate(T_jo_ko.begin(), T_jo_ko.end(), 0.0);
}
void MP2_F12_VBX::calculate_bx_t_fb_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  direct_0_pair_4_one_ints[0] += calculate_bx_t_fb_4e_help(traces.ok12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);
  direct_0_pair_4_one_ints[1] += calculate_bx_t_fb_4e_help(traces.ov12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);

  for (int jo = 0; jo < electron_list->size(); ++jo) {
    for (int ko = 0; ko < electron_list->size(); ++ko) {
      T_jo_ko[jo * traces.electrons + ko] = traces.op12[jo * traces.electrons + ko] * electron_list->inverse_weight[ko] * electron_list->inverse_weight[jo];
    }
  }
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
        std::array<double, 2> t_jo{0.0, 0.0};
        for (int lo = 0; lo < electron_list->size(); ++lo) {
          T_io_jo[lo] = T_jo_ko[io * traces.electrons + lo] * traces.ok12[jo * traces.electrons + lo];
          T_io_ko[lo] = T_jo_ko[io * traces.electrons + lo] * traces.ov12[jo * traces.electrons + lo];
        }
        for (int ko = 0; ko < electron_list->size(); ++ko) {
            std::array<double, 2> t_ko{0.0, 0.0};
            for (int lo = 0; lo < electron_list->size(); ++lo) {
                t_ko[0] += correlation_factor->f12o[ko * traces.electrons + lo] * T_io_jo[lo];
                t_ko[1] += correlation_factor->f12o[ko * traces.electrons + lo] * T_io_ko[lo];
            }
            t_jo[0] += t_ko[0] * T_jo_ko[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko];
            t_jo[1] += t_ko[1] * T_jo_ko[jo * traces.electrons + ko] * traces.ov12[io * traces.electrons + ko];
        }
        auto f_b = correlation_factor->f12o_b[io * traces.electrons + jo];
        t_io[0] += t_jo[0] * f_b;
        t_io[1] += t_jo[1] * f_b;
    }
    xchang_0_pair_4_one_ints[0] += t_io[0];
    xchang_0_pair_4_one_ints[1] += t_io[1];
  }
}
void MP2_F12_VBX::calculate_bx_t_fb(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_bx_t_fb_2e(electron_pair_list, electron_list);
  calculate_bx_t_fb_3e(electron_pair_list, electron_list);
  calculate_bx_t_fb_4e(electron_pair_list, electron_list);
}

void MP2_F12_VBX::calculate_bx_t_fc_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    auto f_12 = correlation_factor->f12p[ip];
    auto f_c = correlation_factor->f12p_c[ip];
    direct_1_pair_0_one_ints[1] += f_12 * f_c * (traces.p11[ip] * traces.dp22[ip] - traces.dp11[ip] * traces.p22[ip]) * electron_pair_list->rv[ip];
    xchang_1_pair_0_one_ints[1] += f_12 * f_c * (traces.dp12[ip] * traces.p12[ip] - traces.p12[ip] * traces.dp21[ip]) * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fc_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23[ip * traces.electrons + io] *  traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.dp31[ip][io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];

      t[2] += correlation_factor->f23[ip * traces.electrons + io] * traces.dp32[ip][io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[3] += correlation_factor->f23[ip * traces.electrons + io] *  traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_c = correlation_factor->f12p_c[ip];
    direct_1_pair_1_one_ints[1] += (t[0] * traces.dp22[ip] - t[1] *  traces.p22[ip]) * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[1] += (t[2] *  traces.p12[ip] - t[3] * traces.dp21[ip]) * f_c * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fc_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 8> s{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
          s[0] += correlation_factor->f12o[io * traces.electrons + jo] * traces.dp32[ip][jo]                     * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          s[1] += correlation_factor->f12o[io * traces.electrons + jo] *  traces.p23[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];

          s[2] += correlation_factor->f12o[io * traces.electrons + jo] * traces.dp32[ip][jo]                     * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          s[3] += correlation_factor->f12o[io * traces.electrons + jo] *  traces.p23[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];

          s[4] += correlation_factor->f12o[io * traces.electrons + jo] *  traces.p13[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          s[5] += correlation_factor->f12o[io * traces.electrons + jo] * traces.dp31[ip][jo]                     * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];

          s[6] += correlation_factor->f12o[io * traces.electrons + jo] *  traces.p13[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          s[7] += correlation_factor->f12o[io * traces.electrons + jo] * traces.dp31[ip][jo]                     * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
      }
      t[0] += (s[0] *  traces.p13[ip * traces.electrons + io] - s[1] * traces.dp31[ip][io]) * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] += (s[2] *  traces.p13[ip * traces.electrons + io] - s[3] * traces.dp31[ip][io]) * traces.v13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[2] += (s[4] * traces.dp32[ip][io] - s[5] *  traces.p23[ip * traces.electrons + io]) * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[3] += (s[6] * traces.dp32[ip][io] - s[7] *  traces.p23[ip * traces.electrons + io]) * traces.v13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_c = correlation_factor->f12p_c[ip];
    direct_1_pair_2_one_ints[1] += t[0] * f_c * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[3] += t[1] * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[1] += t[2] * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[3] += t[3] * f_c * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_t_fc(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_bx_t_fc_2e(electron_pair_list, electron_list);
  calculate_bx_t_fc_3e(electron_pair_list, electron_list);
  calculate_bx_t_fc_4e(electron_pair_list, electron_list);
}

void MP2_F12_VBX::calculate_bx_t_fd_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
      t_io[0] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op11[io] * traces.ds_p22[io][jo] - traces.ds_p11[io][jo] * traces.op11[jo]) * electron_list->inverse_weight[jo];
      t_io[1] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op12[io * traces.electrons + jo] * traces.ds_p12[io][jo] - traces.ds_p21[io][jo] * traces.op12[io * traces.electrons + jo]) * electron_list->inverse_weight[jo];
    }
    direct_0_pair_2_one_ints[1] += t_io[0] * electron_list->inverse_weight[io];
    xchang_0_pair_2_one_ints[1] += t_io[1] * electron_list->inverse_weight[io];
  }
}
void MP2_F12_VBX::calculate_bx_t_fd_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            t_jo[0] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.op12[io * traces.electrons + ko]       * electron_list->inverse_weight[ko];
            t_jo[1] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.ds_p31[io][jo][ko] * electron_list->inverse_weight[ko];
            t_jo[2] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.ds_p32[io][jo][ko] * electron_list->inverse_weight[ko];
            t_jo[3] += correlation_factor->f12o[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko] * traces.op12[jo * traces.electrons + ko]       * electron_list->inverse_weight[ko];
          }
        }
        auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
        t_io[0] += (t_jo[0] * traces.ds_p22[io][jo]   - t_jo[1] * traces.op11[jo]      ) * f_d * electron_list->inverse_weight[jo];
        t_io[1] += (t_jo[2] * traces.op12[io * traces.electrons + jo]     - t_jo[3] * traces.ds_p21[io][jo]) * f_d * electron_list->inverse_weight[jo];
      }
    }
    direct_0_pair_3_one_ints[1] += t_io[0] * electron_list->inverse_weight[io];
    xchang_0_pair_3_one_ints[1] += t_io[1] * electron_list->inverse_weight[io];
  }
}
void MP2_F12_VBX::calculate_bx_t_fd_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
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
              t_ko[0] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p32[io][jo][lo]                     * traces.ok12[jo * traces.electrons + lo] * wgt;
              t_ko[1] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[jo * traces.electrons + lo]       * traces.ok12[jo * traces.electrons + lo] * wgt;

              t_ko[2] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p32[io][jo][lo]                     * traces.ov12[jo * traces.electrons + lo] * wgt;
              t_ko[3] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[jo * traces.electrons + lo]       * traces.ov12[jo * traces.electrons + lo] * wgt;

              t_ko[4] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[io * traces.electrons + lo]       * traces.ok12[jo * traces.electrons + lo] * wgt;
              t_ko[5] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p31[io][jo][lo]                     * traces.ok12[jo * traces.electrons + lo] * wgt;

              t_ko[6] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.op12[io * traces.electrons + lo]       * traces.ov12[jo * traces.electrons + lo] * wgt;
              t_ko[7] += correlation_factor->f12o[ko * traces.electrons + lo] * traces.ds_p31[io][jo][lo]                     * traces.ov12[jo * traces.electrons + lo] * wgt;
            }
          }
          t_jo[0] += (t_ko[0] * traces.op12[io * traces.electrons + ko] - t_ko[1] * traces.ds_p31[io][jo][ko]) * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[1] += (t_ko[2] * traces.op12[io * traces.electrons + ko] - t_ko[3] * traces.ds_p31[io][jo][ko]) * traces.ov12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[2] += (t_ko[4] * traces.ds_p32[io][jo][ko]              - t_ko[5] * traces.op12[jo * traces.electrons + ko]) * traces.ok12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
          t_jo[3] += (t_ko[6] * traces.ds_p32[io][jo][ko]              - t_ko[7] * traces.op12[jo * traces.electrons + ko]) * traces.ov12[io * traces.electrons + ko] * electron_list->inverse_weight[ko];
        }
      }
      auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo];
      t_io[0] += t_jo[0] * f_d * electron_list->inverse_weight[jo];
      t_io[1] += t_jo[1] * f_d * electron_list->inverse_weight[jo];
      t_io[2] += t_jo[2] * f_d * electron_list->inverse_weight[jo];
      t_io[3] += t_jo[3] * f_d * electron_list->inverse_weight[jo];
    }
    direct_0_pair_4_one_ints[2] += t_io[0] * electron_list->inverse_weight[io];
    direct_0_pair_4_one_ints[3] += t_io[1] * electron_list->inverse_weight[io];
    xchang_0_pair_4_one_ints[2] += t_io[2] * electron_list->inverse_weight[io];
    xchang_0_pair_4_one_ints[3] += t_io[3] * electron_list->inverse_weight[io];
  }
}
void MP2_F12_VBX::calculate_bx_t_fd(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_bx_t_fd_2e(electron_pair_list, electron_list);
  calculate_bx_t_fd_3e(electron_pair_list, electron_list);
  calculate_bx_t_fd_4e(electron_pair_list, electron_list);
}

void MP2_F12_VBX::calculate_bx_k_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23[ip * traces.electrons + io] * (correlation_factor->f23[ip * traces.electrons + io] - correlation_factor->f13[ip * traces.electrons + io]) * traces.op11[io] * electron_list->inverse_weight[io];
      t[1] += correlation_factor->f23[ip * traces.electrons + io] * (correlation_factor->f23[ip * traces.electrons + io] - correlation_factor->f13[ip * traces.electrons + io]) * traces.p13[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    direct_1_pair_1_one_ints[2] += t[0] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[2] += t[1] * traces.k12[ip] * electron_pair_list->rv[ip];
  }
}
void MP2_F12_VBX::calculate_bx_k_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::transform(correlation_factor->f13.begin(), correlation_factor->f13.end(), correlation_factor->f23.begin(), T_ip_jo.begin(), std::minus<>());
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        s[0] += correlation_factor->f12o[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op11[jo] * electron_list->inverse_weight[jo];
        s[1] += correlation_factor->f12o[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[2] += correlation_factor->f23[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.ok12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
        s[3] += correlation_factor->f23[ip * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo]  * traces.ok12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
      }
      t[0] += s[0] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[1] += s[1] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[2] -= s[2] * T_ip_jo[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[3] -= s[3] * T_ip_jo[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    direct_1_pair_2_one_ints[4] += t[0] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[4] += t[1] * traces.k12[ip] * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[5] += t[2] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[5] += t[3] * traces.k12[ip] * electron_pair_list->rv[ip];
  }
}
double MP2_F12_VBX::calculate_bx_k_5e_help(
    const std::vector<double>& S_ip_io_1, const std::vector<double>& S_ip_io_2,
    const std::vector<double>& S_jo_ko_1, const std::vector<double>& S_jo_ko_2,
    const std::vector<double>& S_io_ko, const std::vector<double>& S_ip_jo,
    const std::vector<double>& weight,
    size_t size, size_t size_ep) {
  std::transform(S_ip_io_1.begin(), S_ip_io_1.end(), S_ip_io_2.begin(), T_ip_io.begin(), std::multiplies<>());
  std::transform(S_jo_ko_1.begin(), S_jo_ko_1.end(), S_jo_ko_2.begin(), T_jo_ko.begin(), std::multiplies<>());

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      size, size, size_ep,
      1.0,
      S_ip_jo.data(), size,
      T_ip_io.data(), size,
      0.0,
      T_io_jo.data(), size);

  cblas_dscal(size, 0.0, T_io_jo.data(), size+1);

  for (int jo = 0; jo < size; ++jo) {
    for (int ko = 0; ko < size; ++ko) {
      T_jo_ko[jo * size + ko] *= weight[ko];
    }
  }

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      size, size, size_ep,
      1.0,
      T_jo_ko.data(), size,
      S_io_ko.data(), size,
      0.0,
      T_io_ko.data(), size);


  std::transform(T_io_jo.begin(), T_io_jo.end(), T_io_ko.begin(), T_io_jo.begin(), std::multiplies<>());

  double en = 0.0;
  for (int io = 0; io < size; ++io) {
    double t = 0.0;
    for (int jo = 0; jo < size; ++jo) {
      t += T_io_jo[io * size + jo] * weight[jo];
    }
    en += t * weight[io];
  }
  return en;
}
void MP2_F12_VBX::calculate_bx_k_5e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::transform(electron_pair_list->rv.begin(), electron_pair_list->rv.end(), traces.k12.begin(), T_ip.begin(), std::multiplies<>());
  for (int ip = 0, idx = 0; ip < electron_pair_list->size(); ++ip) {
    for (int jo = 0; jo < electron_list->size(); ++jo, ++idx) {
      T_ip_jo[idx] = (correlation_factor->f13[ip * traces.electrons + jo] - correlation_factor->f23[ip * traces.electrons + jo]) * T_ip[ip];
    }
  }

  direct_1_pair_3_one_ints[0] += calculate_bx_k_5e_help(traces.k13, traces.p23, traces.ok12,traces.op12, correlation_factor->f12o, T_ip_jo, 
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());
  direct_1_pair_3_one_ints[0] -= calculate_bx_k_5e_help(traces.v13, traces.p23, traces.ov12,traces.op12, correlation_factor->f12o, T_ip_jo, 
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());

  std::transform(correlation_factor->f13.begin(), correlation_factor->f13.end(), correlation_factor->f23.begin(), T_ip_jo.begin(), std::minus<>());
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      T_ip_jo[ip * traces.electrons + jo] *= electron_pair_list->rv[ip] * electron_list->inverse_weight[jo];
    }
  }
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int ko = 0; ko < electron_list->size(); ++ko) {
      T_io_ko[io * traces.electrons + ko] = correlation_factor->f12o[io * traces.electrons + ko] * electron_list->inverse_weight[io] * electron_list->inverse_weight[ko];
    }
  }

  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 2> t_i{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 2> t_j{0.0, 0.0};
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        T_ip_io[ko] = T_io_ko[io * traces.electrons + ko] * traces.p23[ip * traces.electrons + ko];
      }
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        std::array<double, 2> t_k{0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          t_k[0] += T_ip_io[ko] * traces.ok12[jo * traces.electrons + ko];
          t_k[1] += T_ip_io[ko] * traces.ov12[jo * traces.electrons + ko];
        }
        t_j[0] += t_k[0] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo]; 
        t_j[1] += t_k[1] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo]; 
      }
      t_i[0] += t_j[0] * traces.k13[ip * traces.electrons + io];
      t_i[1] += t_j[1] * traces.v13[ip * traces.electrons + io];
    }
    xchang_1_pair_3_one_ints[0] += t_i[0] * traces.k12[ip];
    xchang_1_pair_3_one_ints[0] -= t_i[1] * traces.k12[ip];
  }
}
void MP2_F12_VBX::calculate_bx_k(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  calculate_bx_k_3e(electron_pair_list, electron_list);
  calculate_bx_k_4e(electron_pair_list, electron_list);
  calculate_bx_k_5e(electron_pair_list, electron_list);
}

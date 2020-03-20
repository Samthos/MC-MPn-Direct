#include <algorithm>
#include <iostream>
#include <numeric>

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

void MP2_F12_V::calculate_v(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);

  std::array<double, 2> v_1_pair_0_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    auto f_12 = correlation_factor->calculate_f12(electron_pair_list->r12[ip]);
    v_1_pair_0_one_ints[0] = v_1_pair_0_one_ints[0] + f_12 * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    v_1_pair_0_one_ints[1] = v_1_pair_0_one_ints[1] + f_12 * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_0_one_int : v_1_pair_0_one_ints) {
    v_1_pair_0_one_int *= nsamp_pair;
  }

  std::array<double, 2> v_1_pair_1_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] = t[0] + correlation_factor->f23p[ip][io] * traces.p13[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t[1] = t[1] + correlation_factor->f23p[ip][io] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
    }
    v_1_pair_1_one_ints[0] = v_1_pair_1_one_ints[0] + t[0] * traces.p22[ip] * electron_pair_list->rv[ip];
    v_1_pair_1_one_ints[1] = v_1_pair_1_one_ints[1] + t[1] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_1_one_int : v_1_pair_1_one_ints) {
    v_1_pair_1_one_int *= nsamp_pair * nsamp_one_1;
  }

  std::array<double, 2> v_1_pair_2_one_ints{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          s[0] = s[0] + correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];
          s[1] = s[1] + correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];
          s[2] = s[2] + correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
          s[3] = s[3] + correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
        }
      }
      t[0] = t[0] + traces.p13[ip][io] * (s[0] * traces.k13[ip][io] - s[2] * traces.v13[ip][io]) / electron_list->weight[io];
      t[1] = t[1] + traces.p23[ip][io] * (s[1] * traces.k13[ip][io] - s[3] * traces.v13[ip][io]) / electron_list->weight[io];
    }
    v_1_pair_2_one_ints[0] = v_1_pair_2_one_ints[0] + t[0] * electron_pair_list->rv[ip];
    v_1_pair_2_one_ints[1] = v_1_pair_2_one_ints[1] + t[1] * electron_pair_list->rv[ip];
  }
  for (double & v_1_pair_2_one_int : v_1_pair_2_one_ints) {
    v_1_pair_2_one_int *= nsamp_pair * nsamp_one_2;
  }

  emp +=   c1 * (v_1_pair_0_one_ints[0] + v_1_pair_2_one_ints[0] - 2 * v_1_pair_1_one_ints[0])
        + c2 * (v_1_pair_0_one_ints[1] + v_1_pair_2_one_ints[1] - 2 * v_1_pair_1_one_ints[1]);
}

MP2_F12_VBX::MP2_F12_VBX(const IOPs& iops, const Basis& basis) : MP2_F12_V(iops, basis, "f12_VBX") {
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
  /* Due to the symetry of the integrals in calculate_bx_t_fd, calculate_bx_t_fd always evalates to zero analatically.
   * if (!correlation_factor->f12_d_is_zero()) {
   *   traces.update_bx_fd_traces(wavefunctions, electron_list);
   *   calculate_bx_t_fd(electron_pair_list, electron_list);
   * }
   */

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

void MP2_F12_VBX::calculate_bx_t_fa(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list->r12[ip]);
    auto f_12 = correlation_factor->calculate_f12(electron_pair_list->r12[ip]);
    direct_1_pair_0_one_ints[0] += f_12 * f_a * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_0_one_ints[0] += f_12 * f_a * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23p[ip][io] * traces.p13[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t[1] += correlation_factor->f23p[ip][io] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
    }
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list->r12[ip]);
    direct_1_pair_1_one_ints[0] += t[0] * f_a * traces.p22[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[0] += t[1] * f_a * traces.p12[ip] * electron_pair_list->rv[ip];
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for(int jo = 0; jo < electron_list->size();++jo) {
        if (jo != io) {
          s[0] += correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);
          s[1] += correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);
          s[2] += correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);
          s[3] += correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);
        }
      }
      t[0] += s[0] * traces.p13[ip][io] * traces.k13[ip][io] / (electron_list->weight[io]);
      t[1] += s[1] * traces.p13[ip][io] * traces.v13[ip][io] / (electron_list->weight[io]);
      t[2] += s[2] * traces.p23[ip][io] * traces.k13[ip][io] / (electron_list->weight[io]);
      t[3] += s[3] * traces.p23[ip][io] * traces.v13[ip][io] / (electron_list->weight[io]);
    }
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list->r12[ip]);
    direct_1_pair_2_one_ints[0] += t[0] * f_a * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[2] += t[1] * f_a * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[0] += t[2] * f_a * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[2] += t[3] * f_a * electron_pair_list->rv[ip];
  }
}

void MP2_F12_VBX::calculate_bx_t_fb(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += f_b * correlation_factor->f12o[io][jo] * traces.op11[jo] / electron_list->weight[jo];
        t_io[1] += f_b * correlation_factor->f12o[io][jo] * traces.op12[io][jo] * traces.op12[io][jo] / electron_list->weight[jo];
      }
    }
    direct_0_pair_2_one_ints[0] += t_io[0] * traces.op11[io] / electron_list->weight[io];
    xchang_0_pair_2_one_ints[0] += t_io[1] / electron_list->weight[io];
  }

  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 2> t_jo{0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            t_jo[0] += correlation_factor->f12o[jo][ko] * traces.op12[io][ko] * traces.ok12[io][ko] / electron_list->weight[ko];
            t_jo[1] += correlation_factor->f12o[jo][ko] * traces.op12[jo][ko] * traces.ok12[io][ko] / electron_list->weight[ko];
          }
        }
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += t_jo[0] * f_b * traces.op11[jo]     / electron_list->weight[jo];
        t_io[1] += t_jo[1] * f_b * traces.op12[io][jo] / electron_list->weight[jo];
      }
    }
    direct_0_pair_3_one_ints[0] += t_io[0] / electron_list->weight[io];
    xchang_0_pair_3_one_ints[0] += t_io[1] / electron_list->weight[io];
  }

  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0, 0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            std::array<double, 4> t_ko{0.0, 0.0, 0.0, 0.0};
            for (int lo = 0; lo < electron_list->size(); ++lo) {
              if (lo != ko && lo != jo && lo != io) {
                t_ko[0] += correlation_factor->f12o[ko][lo] * traces.op12[jo][lo] * traces.ok12[jo][lo] / (electron_list->weight[lo]);
                t_ko[1] += correlation_factor->f12o[ko][lo] * traces.op12[jo][lo] * traces.ov12[jo][lo] / (electron_list->weight[lo]);
                t_ko[2] += correlation_factor->f12o[ko][lo] * traces.op12[io][lo] * traces.ok12[jo][lo] / (electron_list->weight[lo]);
                t_ko[3] += correlation_factor->f12o[ko][lo] * traces.op12[io][lo] * traces.ov12[jo][lo] / (electron_list->weight[lo]);
              }
            }
            t_jo[0] += t_ko[0] * traces.op12[io][ko] * traces.ok12[io][ko] / electron_list->weight[ko];
            t_jo[1] += t_ko[1] * traces.op12[io][ko] * traces.ov12[io][ko] / electron_list->weight[ko];
            t_jo[2] += t_ko[2] * traces.op12[jo][ko] * traces.ok12[io][ko] / electron_list->weight[ko];
            t_jo[3] += t_ko[3] * traces.op12[jo][ko] * traces.ov12[io][ko] / electron_list->weight[ko];
          }
        }
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += t_jo[0] * f_b / electron_list->weight[jo];
        t_io[1] += t_jo[1] * f_b / electron_list->weight[jo];
        t_io[2] += t_jo[2] * f_b / electron_list->weight[jo];
        t_io[3] += t_jo[3] * f_b / electron_list->weight[jo];
      }
    }
    direct_0_pair_4_one_ints[0] += t_io[0] / electron_list->weight[io];
    direct_0_pair_4_one_ints[1] += t_io[1] / electron_list->weight[io];
    xchang_0_pair_4_one_ints[0] += t_io[2] / electron_list->weight[io];
    xchang_0_pair_4_one_ints[1] += t_io[3] / electron_list->weight[io];
  }
}

void MP2_F12_VBX::calculate_bx_t_fc(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto f_12 = correlation_factor->calculate_f12(electron_pair_list->r12[ip]);
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list->r12[ip]);
    direct_1_pair_0_one_ints[1] += f_12 * f_c * (traces.p11[ip] * traces.dp22[ip] - traces.dp11[ip] * traces.p22[ip]) * electron_pair_list->rv[ip];
    xchang_1_pair_0_one_ints[1] += f_12 * f_c * (traces.dp12[ip] * traces.p12[ip] - traces.p12[ip] * traces.dp21[ip]) * electron_pair_list->rv[ip];
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23p[ip][io] *  traces.p13[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t[1] += correlation_factor->f23p[ip][io] * traces.dp31[ip][io] * traces.k13[ip][io] / electron_list->weight[io];

      t[2] += correlation_factor->f23p[ip][io] * traces.dp32[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t[3] += correlation_factor->f23p[ip][io] *  traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
    }
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list->r12[ip]);
    direct_1_pair_1_one_ints[1] += (t[0] * traces.dp22[ip] - t[1] *  traces.p22[ip]) * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[1] += (t[2] *  traces.p12[ip] - t[3] * traces.dp21[ip]) * f_c * electron_pair_list->rv[ip];
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 8> s{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          s[0] += correlation_factor->f12o[io][jo] * traces.dp32[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);
          s[1] += correlation_factor->f12o[io][jo] *  traces.p23[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);

          s[2] += correlation_factor->f12o[io][jo] * traces.dp32[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);
          s[3] += correlation_factor->f12o[io][jo] *  traces.p23[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);

          s[4] += correlation_factor->f12o[io][jo] *  traces.p13[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);
          s[5] += correlation_factor->f12o[io][jo] * traces.dp31[ip][jo] * traces.k23[ip][jo] / (electron_list->weight[jo]);

          s[6] += correlation_factor->f12o[io][jo] *  traces.p13[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);
          s[7] += correlation_factor->f12o[io][jo] * traces.dp31[ip][jo] * traces.v23[ip][jo] / (electron_list->weight[jo]);
        }
      }
      t[0] += (s[0] *  traces.p13[ip][io] - s[1] * traces.dp31[ip][io]) * traces.k13[ip][io] / (electron_list->weight[io]);
      t[1] += (s[2] *  traces.p13[ip][io] - s[3] * traces.dp31[ip][io]) * traces.v13[ip][io] / (electron_list->weight[io]);
      t[2] += (s[4] * traces.dp32[ip][io] - s[5] *  traces.p23[ip][io]) * traces.k13[ip][io] / (electron_list->weight[io]);
      t[3] += (s[6] * traces.dp32[ip][io] - s[7] *  traces.p23[ip][io]) * traces.v13[ip][io] / (electron_list->weight[io]);
    }
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list->r12[ip]);
    direct_1_pair_2_one_ints[1] += t[0] * f_c * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[3] += t[1] * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[1] += t[2] * f_c * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[3] += t[3] * f_c * electron_pair_list->rv[ip];
  }
}

void MP2_F12_VBX::calculate_bx_t_fd(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += f_d * correlation_factor->f12o[io][jo] * traces.ds_p22[io][jo] / (electron_list->weight[jo]);
        t_io[1] += f_d * correlation_factor->f12o[io][jo] * traces.op12[io][jo] * traces.ds_p12[io][jo] / (electron_list->weight[jo]);
      }
    }
    direct_0_pair_2_one_ints[1] += t_io[0] * traces.op11[io] / electron_list->weight[io];
    xchang_0_pair_2_one_ints[1] += t_io[1] / electron_list->weight[io];
  }
  direct_0_pair_2_one_ints[1] *= 2.0;
  xchang_0_pair_2_one_ints[1] *= 2.0;

  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            t_jo[0] += correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.op12[io][ko]       / electron_list->weight[ko];
            t_jo[1] += correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.ds_p31[io][jo][ko] / electron_list->weight[ko];
            t_jo[2] += correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.ds_p32[io][jo][ko] / electron_list->weight[ko];
            t_jo[3] += correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.op12[jo][ko]       / electron_list->weight[ko];
          }
        }
        auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += (t_jo[0] * traces.ds_p22[io][jo]   - t_jo[1] * traces.op11[jo]      ) * f_d / electron_list->weight[jo];
        t_io[1] += (t_jo[2] * traces.op12[io][jo]     - t_jo[3] * traces.ds_p21[io][jo]) * f_d / electron_list->weight[jo];
      }
    }
    direct_0_pair_3_one_ints[1] += t_io[0] / electron_list->weight[io];
    xchang_0_pair_3_one_ints[1] += t_io[1] / electron_list->weight[io];
  }

  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 4> t_io{0.0, 0.0, 0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            std::array<double, 8> t_ko{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int lo = 0; lo < electron_list->size(); ++lo) {
              if (lo != ko && lo != jo && lo != io) {
                t_ko[0] += traces.ds_p32[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list->weight[lo];
                t_ko[1] += traces.op12[jo][lo]       * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list->weight[lo];

                t_ko[2] += traces.ds_p32[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list->weight[lo];
                t_ko[3] += traces.op12[jo][lo]       * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list->weight[lo];

                t_ko[4] += traces.op12[io][lo]       * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list->weight[lo];
                t_ko[5] += traces.ds_p31[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list->weight[lo];

                t_ko[6] += traces.op12[io][lo]       * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list->weight[lo];
                t_ko[7] += traces.ds_p31[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list->weight[lo];
              }
            }
            t_jo[0] += (t_ko[0] * traces.op12[io][ko] - t_ko[1] * traces.ds_p31[io][jo][ko]) * traces.ok12[io][ko] / electron_list->weight[ko];
            t_jo[1] += (t_ko[2] * traces.op12[io][ko] - t_ko[3] * traces.ds_p31[io][jo][ko]) * traces.ov12[io][ko] / electron_list->weight[ko];
            t_jo[2] += (t_ko[4] * traces.ds_p32[io][jo][ko] - t_ko[5] * traces.op12[jo][ko]) * traces.ok12[io][ko] / electron_list->weight[ko];
            t_jo[3] += (t_ko[6] * traces.ds_p32[io][jo][ko] - t_ko[7] * traces.op12[jo][ko]) * traces.ov12[io][ko] / electron_list->weight[ko];
          }
        }
        auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
        t_io[0] += t_jo[0] * f_d / electron_list->weight[jo];
        t_io[1] += t_jo[1] * f_d / electron_list->weight[jo];
        t_io[2] += t_jo[2] * f_d / electron_list->weight[jo];
        t_io[3] += t_jo[3] * f_d / electron_list->weight[jo];
      }
    }
    direct_0_pair_4_one_ints[2] += t_io[0] / electron_list->weight[io];
    direct_0_pair_4_one_ints[3] += t_io[1] / electron_list->weight[io];
    xchang_0_pair_4_one_ints[2] += t_io[2] / electron_list->weight[io];
    xchang_0_pair_4_one_ints[3] += t_io[3] / electron_list->weight[io];
  }
}

void MP2_F12_VBX::calculate_bx_k(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] += correlation_factor->f23p[ip][io] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.op11[io] / electron_list->weight[io];
      t[1] += correlation_factor->f23p[ip][io] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.p13[ip][io] * traces.p23[ip][io] / electron_list->weight[io];
    }
    direct_1_pair_1_one_ints[2] += t[0] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_1_one_ints[2] += t[1] * traces.k12[ip] * electron_pair_list->rv[ip];
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          s[0] += correlation_factor->f12o[io][jo] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op11[jo] / electron_list->weight[jo];
          s[1] += correlation_factor->f12o[io][jo] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op12[io][jo] * traces.p23[ip][jo] / electron_list->weight[jo];
          s[2] += correlation_factor->f23p[ip][jo] * traces.op12[io][jo] * traces.ok12[io][jo] / electron_list->weight[jo];
          s[3] += correlation_factor->f23p[ip][jo] * traces.p13[ip][jo]  * traces.ok12[io][jo] / electron_list->weight[jo];
        }
      }
      t[0] += s[0] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t[1] += s[1] * traces.k13[ip][io] / electron_list->weight[io];
      t[2] += s[2] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) / electron_list->weight[io];
      t[3] += s[3] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.p23[ip][io] / electron_list->weight[io];
    }
    direct_1_pair_2_one_ints[4] += t[0] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[4] += t[1] * traces.k12[ip] * electron_pair_list->rv[ip];
    direct_1_pair_2_one_ints[5] += t[2] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_2_one_ints[5] += t[3] * traces.k12[ip] * electron_pair_list->rv[ip];
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t_i{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> t_j{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          std::array<double, 4> t_k{0.0, 0.0, 0.0, 0.0};
          for (int ko = 0; ko < electron_list->size(); ++ko) {
            if (ko != jo && ko != io) {
            // 
            // save for strength reduction in future
            // direct_1_pair_3_one_ints[0] = direct_1_pair_3_one_ints[0] +
            //     correlation_factor->f12o[jo][ko] *
            //         (correlation_factor->f13p[ip][io] - correlation_factor->f23p[ip][io]) *
            //         traces.k12[ip] *
            //         (sum_ok12[io][jo] * traces.k13[ip][ko] - traces.ov12[io][jo] * traces.v13[ip][ko]) *
            //         traces.op12[io][jo] *
            //         traces.p23[ip][ko] * electron_pair_list->rv[ip] / (electron_list->weight[io] * electron_list->weight[jo] * electron_list->weight[ko]);
            //
            t_k[0] += correlation_factor->f12o[io][ko] * traces.op12[jo][ko] * traces.ok12[jo][ko] / (electron_list->weight[ko]);
            t_k[1] += correlation_factor->f12o[io][ko] * traces.op12[jo][ko] * traces.ov12[jo][ko] / (electron_list->weight[ko]);
                                                                            
            t_k[2] += correlation_factor->f12o[io][ko] * traces.p23[ip][ko] * traces.ok12[jo][ko] / (electron_list->weight[ko]);
            t_k[3] += correlation_factor->f12o[io][ko] * traces.p23[ip][ko] * traces.ov12[jo][ko] / (electron_list->weight[ko]);
            }
          }
          t_j[0] += t_k[0] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) / electron_list->weight[jo]; 
          t_j[1] += t_k[1] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) / electron_list->weight[jo]; 
          t_j[2] += t_k[2] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op12[io][jo] / electron_list->weight[jo]; 
          t_j[3] += t_k[3] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op12[io][jo] / electron_list->weight[jo]; 
        }
      }

      t_i[0] += t_j[0] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      t_i[1] += t_j[1] * traces.p23[ip][io] * traces.v13[ip][io] / electron_list->weight[io];
      t_i[2] += t_j[2] * traces.k13[ip][io] / electron_list->weight[io];
      t_i[3] += t_j[3] * traces.v13[ip][io] / electron_list->weight[io];
    }
    direct_1_pair_3_one_ints[0] += t_i[0] * traces.k12[ip] * electron_pair_list->rv[ip];
    direct_1_pair_3_one_ints[0] -= t_i[1] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_3_one_ints[0] += t_i[2] * traces.k12[ip] * electron_pair_list->rv[ip];
    xchang_1_pair_3_one_ints[0] -= t_i[3] * traces.k12[ip] * electron_pair_list->rv[ip];
  }
}

#include <algorithm>
#include <iostream>
#include <numeric>

#include "mp2f12_var.h"

double MP2F12_V_Engine::calculate_v(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  traces.update_v(electron_pair_psi1, electron_pair_psi2, electron_psi);
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
  /*
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
  */

  std::array<double, 2> v_1_pair_2_one_ints{0.0, 0.0};
  /*
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
  */

  auto eV =   2 * c1 * (v_1_pair_0_one_ints[0] + v_1_pair_2_one_ints[0] - 2 * v_1_pair_1_one_ints[0])
            + 2 * c2 * (v_1_pair_0_one_ints[1] + v_1_pair_2_one_ints[1] - 2 * v_1_pair_1_one_ints[1]);
  return eV;
}

/*
void MP2F12_VBX_Engine::zero() {
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

void MP2F12_VBX_Engine::calculate_bx_t_fa(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list[ip].r12);
    auto f_12 = correlation_factor->calculate_f12(electron_pair_list[ip].r12);
    direct_1_pair_0_one_ints[0] = direct_1_pair_0_one_ints[0] + f_12 * f_a * traces.p11[ip] * traces.p22[ip] * electron_pair_list[ip].rv;
    xchang_1_pair_0_one_ints[0] = xchang_1_pair_0_one_ints[0] + f_12 * f_a * traces.p12[ip] * traces.p12[ip] * electron_pair_list[ip].rv;
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] = t[0] + correlation_factor->f23p[ip][io] * traces.p13[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
      t[1] = t[1] + correlation_factor->f23p[ip][io] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
    }
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list[ip].r12);
    direct_1_pair_1_one_ints[0] = direct_1_pair_1_one_ints[0] + t[0] * f_a * traces.p22[ip] * electron_pair_list[ip].rv;
    xchang_1_pair_1_one_ints[0] = xchang_1_pair_1_one_ints[0] + t[1] * f_a * traces.p12[ip] * electron_pair_list[ip].rv;
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for(int jo = 0; jo < electron_list->size();++jo) {
        if (jo != io) {
          s[0] = s[0] + correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[1] = s[1] + correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
          s[2] = s[2] + correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[3] = s[3] + correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
        }
      }
      t[0] = t[0] + s[0] * traces.p13[ip][io] * traces.k13[ip][io] / (electron_list[io].wgt);
      t[1] = t[1] + s[1] * traces.p13[ip][io] * traces.v13[ip][io] / (electron_list[io].wgt);
      t[2] = t[2] + s[2] * traces.p23[ip][io] * traces.k13[ip][io] / (electron_list[io].wgt);
      t[3] = t[3] + s[3] * traces.p23[ip][io] * traces.v13[ip][io] / (electron_list[io].wgt);
    }
    auto f_a = correlation_factor->calculate_f12_a(electron_pair_list[ip].r12);
    direct_1_pair_2_one_ints[0] = direct_1_pair_2_one_ints[0] + t[0] * f_a * electron_pair_list[ip].rv;
    direct_1_pair_2_one_ints[2] = direct_1_pair_2_one_ints[2] + t[1] * f_a * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[0] = xchang_1_pair_2_one_ints[0] + t[2] * f_a * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[2] = xchang_1_pair_2_one_ints[2] + t[3] * f_a * electron_pair_list[ip].rv;
  }
}

void MP2F12_VBX_Engine::calculate_bx_t_fb(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] = t_io[0] + f_b * correlation_factor->f12o[io][jo] * traces.op11[jo] / electron_list[jo].wgt;
        t_io[1] = t_io[1] + f_b * correlation_factor->f12o[io][jo] * traces.op12[io][jo] * traces.op12[io][jo] / electron_list[jo].wgt;
      }
    }
    direct_0_pair_2_one_ints[0] = direct_0_pair_2_one_ints[0] + t_io[0] * traces.op11[io] / electron_list[io].wgt;
    xchang_0_pair_2_one_ints[0] = xchang_0_pair_2_one_ints[0] + t_io[1] / electron_list[io].wgt;
  }

  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        std::array<double, 2> t_jo{0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            t_jo[0] = t_jo[0] + correlation_factor->f12o[jo][ko] * traces.op12[io][ko] * traces.ok12[io][ko] / electron_list[ko].wgt;
            t_jo[1] = t_jo[1] + correlation_factor->f12o[jo][ko] * traces.op12[jo][ko] * traces.ok12[io][ko] / electron_list[ko].wgt;
          }
        }
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] = t_io[0] + t_jo[0] * f_b * traces.op11[jo]     / electron_list[jo].wgt;
        t_io[1] = t_io[1] + t_jo[1] * f_b * traces.op12[io][jo] / electron_list[jo].wgt;
      }
    }
    direct_0_pair_3_one_ints[0] = direct_0_pair_3_one_ints[0] + t_io[0] / electron_list[io].wgt;
    xchang_0_pair_3_one_ints[0] = xchang_0_pair_3_one_ints[0] + t_io[1] / electron_list[io].wgt;
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
                t_ko[0] = t_ko[0] + correlation_factor->f12o[ko][lo] * traces.op12[jo][lo] * traces.ok12[jo][lo] / (electron_list[lo].wgt);
                t_ko[1] = t_ko[1] + correlation_factor->f12o[ko][lo] * traces.op12[jo][lo] * traces.ov12[jo][lo] / (electron_list[lo].wgt);
                t_ko[2] = t_ko[2] + correlation_factor->f12o[ko][lo] * traces.op12[io][lo] * traces.ok12[jo][lo] / (electron_list[lo].wgt);
                t_ko[3] = t_ko[3] + correlation_factor->f12o[ko][lo] * traces.op12[io][lo] * traces.ov12[jo][lo] / (electron_list[lo].wgt);
              }
            }
            t_jo[0] = t_jo[0] + t_ko[0] * traces.op12[io][ko] * traces.ok12[io][ko] / electron_list[ko].wgt;
            t_jo[1] = t_jo[1] + t_ko[1] * traces.op12[io][ko] * traces.ov12[io][ko] / electron_list[ko].wgt;
            t_jo[2] = t_jo[2] + t_ko[2] * traces.op12[jo][ko] * traces.ok12[io][ko] / electron_list[ko].wgt;
            t_jo[3] = t_jo[3] + t_ko[3] * traces.op12[jo][ko] * traces.ov12[io][ko] / electron_list[ko].wgt;
          }
        }
        auto f_b = correlation_factor->calculate_f12_b(correlation_factor->one_e_r12[io][jo]);
        t_io[0] = t_io[0] + t_jo[0] * f_b / electron_list[jo].wgt;
        t_io[1] = t_io[1] + t_jo[1] * f_b / electron_list[jo].wgt;
        t_io[2] = t_io[2] + t_jo[2] * f_b / electron_list[jo].wgt;
        t_io[3] = t_io[3] + t_jo[3] * f_b / electron_list[jo].wgt;
      }
    }
    direct_0_pair_4_one_ints[0] = direct_0_pair_4_one_ints[0] + t_io[0] / electron_list[io].wgt;
    direct_0_pair_4_one_ints[1] = direct_0_pair_4_one_ints[1] + t_io[1] / electron_list[io].wgt;
    xchang_0_pair_4_one_ints[0] = xchang_0_pair_4_one_ints[0] + t_io[2] / electron_list[io].wgt;
    xchang_0_pair_4_one_ints[1] = xchang_0_pair_4_one_ints[1] + t_io[3] / electron_list[io].wgt;
  }
}

void MP2F12_VBX_Engine::calculate_bx_t_fc(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto f_12 = correlation_factor->calculate_f12(electron_pair_list[ip].r12);
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list[ip].r12);
    direct_1_pair_0_one_ints[1] = direct_1_pair_0_one_ints[1] + f_12 * f_c * (traces.p11[ip] * traces.dp22[ip] - traces.dp11[ip] * traces.p22[ip]) * electron_pair_list[ip].rv;
    xchang_1_pair_0_one_ints[1] = xchang_1_pair_0_one_ints[1] + f_12 * f_c * (traces.dp12[ip] * traces.p12[ip] - traces.p12[ip] * traces.dp21[ip]) * electron_pair_list[ip].rv;
  }

  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] = t[0] + correlation_factor->f23p[ip][io] *  traces.p13[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
      t[1] = t[1] + correlation_factor->f23p[ip][io] * traces.dp32[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
      t[2] = t[2] + correlation_factor->f23p[ip][io] * traces.dp31[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
      t[3] = t[3] + correlation_factor->f23p[ip][io] *  traces.p23[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
    }
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list[ip].r12);
    direct_1_pair_1_one_ints[1] = direct_1_pair_1_one_ints[1] + (t[0] * traces.dp22[ip] - t[2] *  traces.p22[ip]) * f_c * electron_pair_list[ip].rv;
    xchang_1_pair_1_one_ints[1] = xchang_1_pair_1_one_ints[1] + (t[1] *  traces.p12[ip] - t[3] * traces.dp21[ip]) * f_c * electron_pair_list[ip].rv;
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 8> s{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          s[0] = s[0] + correlation_factor->f12o[io][jo] * traces.dp32[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[1] = s[1] + correlation_factor->f12o[io][jo] * traces.dp32[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
          s[2] = s[2] + correlation_factor->f12o[io][jo] *  traces.p13[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[3] = s[3] + correlation_factor->f12o[io][jo] *  traces.p13[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
          s[4] = s[4] + correlation_factor->f12o[io][jo] *  traces.p23[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[5] = s[5] + correlation_factor->f12o[io][jo] *  traces.p23[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
          s[6] = s[6] + correlation_factor->f12o[io][jo] * traces.dp31[ip][jo] * traces.k23[ip][jo] / (electron_list[jo].wgt);
          s[7] = s[7] + correlation_factor->f12o[io][jo] * traces.dp31[ip][jo] * traces.v23[ip][jo] / (electron_list[jo].wgt);
        }
      }
      t[0] = t[0] + (s[0] *  traces.p13[ip][io] - s[4] * traces.dp31[ip][io]) * traces.k13[ip][io] / (electron_list[io].wgt);
      t[1] = t[1] + (s[1] *  traces.p13[ip][io] - s[5] * traces.dp31[ip][io]) * traces.v13[ip][io] / (electron_list[io].wgt);
      t[2] = t[2] + (s[2] * traces.dp32[ip][io] - s[6] *  traces.p23[ip][io]) * traces.k13[ip][io] / (electron_list[io].wgt);
      t[3] = t[3] + (s[3] * traces.dp32[ip][io] - s[7] *  traces.p23[ip][io]) * traces.v13[ip][io] / (electron_list[io].wgt);
    }
    auto f_c = correlation_factor->calculate_f12_c(electron_pair_list[ip].r12);
    direct_1_pair_2_one_ints[1] = direct_1_pair_2_one_ints[1] + t[0] * f_c * electron_pair_list[ip].rv;
    direct_1_pair_2_one_ints[3] = direct_1_pair_2_one_ints[3] + t[1] * f_c * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[1] = xchang_1_pair_2_one_ints[1] + t[2] * f_c * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[3] = xchang_1_pair_2_one_ints[3] + t[3] * f_c * electron_pair_list[ip].rv;
  }
}

void MP2F12_VBX_Engine::calculate_bx_t_fd(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  if (!correlation_factor->f12_d_is_zero()) {
    traces.update_bx_fd_traces(electron_list);

    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 2> t_io{0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
          t_io[0] = t_io[0] + f_d * correlation_factor->f12o[io][jo] * traces.ds_p22[io][jo] / (electron_list[jo].wgt);
          t_io[1] = t_io[1] + f_d * correlation_factor->f12o[io][jo] * traces.op12[io][jo] * traces.ds_p12[io][jo] / (electron_list[jo].wgt);
        }
      }
      direct_0_pair_2_one_ints[1] = direct_0_pair_2_one_ints[1] + t_io[0] * traces.op11[io] / electron_list[io].wgt;
      xchang_0_pair_2_one_ints[1] = xchang_0_pair_2_one_ints[1] + t_io[1] / electron_list[io].wgt;
    }
    direct_0_pair_2_one_ints[1] = 2.0 * direct_0_pair_2_one_ints[1];
    xchang_0_pair_2_one_ints[1] = 2.0 * xchang_0_pair_2_one_ints[1];

    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> t_io{0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
          for (int ko = 0; ko < electron_list->size(); ++ko) {
            if (ko != jo && ko != io) {
              t_jo[0] = t_jo[0] + correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.op12[io][ko]   / electron_list[ko].wgt;
              t_jo[1] = t_jo[1] + correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.ds_p31[io][jo][ko] / electron_list[ko].wgt;
              t_jo[2] = t_jo[2] + correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.ds_p32[io][jo][ko] / electron_list[ko].wgt;
              t_jo[3] = t_jo[3] + correlation_factor->f12o[jo][ko] * traces.ok12[io][ko] * traces.op12[jo][ko]   / electron_list[ko].wgt;
            }
          }
          auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
          t_io[0] = t_io[0] + (t_jo[0] * traces.ds_p22[io][jo]   - t_jo[1] * traces.op11[jo]  ) * f_d / electron_list[jo].wgt;
          t_io[1] = t_io[1] + (t_jo[2] * traces.op12[io][jo] - t_jo[3] * traces.ds_p21[io][jo]) * f_d / electron_list[jo].wgt;
        }
      }
      direct_0_pair_3_one_ints[1] = direct_0_pair_3_one_ints[1] + t_io[0] / electron_list[io].wgt;
      xchang_0_pair_3_one_ints[1] = xchang_0_pair_3_one_ints[1] + t_io[1] / electron_list[io].wgt;
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
                  t_ko[0] = t_ko[0] + traces.ds_p32[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list[lo].wgt;
                  t_ko[1] = t_ko[1] + traces.op12[jo][lo]   * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list[lo].wgt;
                  t_ko[2] = t_ko[2] + traces.ds_p32[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list[lo].wgt;
                  t_ko[3] = t_ko[3] + traces.op12[jo][lo]   * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list[lo].wgt;
                  t_ko[4] = t_ko[4] + traces.op12[io][lo]   * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list[lo].wgt;
                  t_ko[5] = t_ko[5] + traces.ds_p31[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ok12[jo][lo] / electron_list[lo].wgt;
                  t_ko[6] = t_ko[6] + traces.op12[io][lo]   * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list[lo].wgt;
                  t_ko[7] = t_ko[7] + traces.ds_p31[io][jo][lo] * correlation_factor->f12o[ko][lo] * traces.ov12[jo][lo] / electron_list[lo].wgt;
                }
              }
              t_jo[0] = t_jo[0] + (t_ko[0] * traces.op12[io][ko] - t_ko[1] * traces.ds_p31[io][jo][ko]) * traces.ok12[io][ko] / electron_list[ko].wgt;
              t_jo[1] = t_jo[1] + (t_ko[2] * traces.op12[io][ko] - t_ko[3] * traces.ds_p31[io][jo][ko]) * traces.ov12[io][ko] / electron_list[ko].wgt;
              t_jo[2] = t_jo[2] + (t_ko[4] * traces.ds_p32[io][jo][ko] - t_ko[5] * traces.op12[jo][ko]) * traces.ok12[io][ko] / electron_list[ko].wgt;
              t_jo[3] = t_jo[3] + (t_ko[6] * traces.ds_p32[io][jo][ko] - t_ko[7] * traces.op12[jo][ko]) * traces.ov12[io][ko] / electron_list[ko].wgt;
            }
          }
          auto f_d = correlation_factor->calculate_f12_d(correlation_factor->one_e_r12[io][jo]);
          t_io[0] = t_io[0] + t_jo[0] * f_d / electron_list[jo].wgt;
          t_io[1] = t_io[1] + t_jo[1] * f_d / electron_list[jo].wgt;
          t_io[2] = t_io[2] + t_jo[2] * f_d / electron_list[jo].wgt;
          t_io[3] = t_io[3] + t_jo[3] * f_d / electron_list[jo].wgt;
        }
      }
      direct_0_pair_4_one_ints[2] = direct_0_pair_4_one_ints[2] + t_io[0] / electron_list[io].wgt;
      direct_0_pair_4_one_ints[3] = direct_0_pair_4_one_ints[3] + t_io[1] / electron_list[io].wgt;
      xchang_0_pair_4_one_ints[2] = xchang_0_pair_4_one_ints[2] + t_io[2] / electron_list[io].wgt;
      xchang_0_pair_4_one_ints[3] = xchang_0_pair_4_one_ints[3] + t_io[3] / electron_list[io].wgt;
    }
  }
}

void MP2F12_VBX_Engine::calculate_bx_k(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 2> t{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t[0] = t[0] + correlation_factor->f23p[ip][io] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.op11[io] / electron_list[io].wgt;
      t[1] = t[1] + correlation_factor->f23p[ip][io] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.p13[ip][io] * traces.p23[ip][io] / electron_list[io].wgt;
    }
    direct_1_pair_1_one_ints[2] = direct_1_pair_1_one_ints[2] + t[0] * traces.p12[ip] * traces.k12[ip] * electron_pair_list[ip].rv;
    xchang_1_pair_1_one_ints[2] = xchang_1_pair_1_one_ints[2] + t[1] * traces.k12[ip] * electron_pair_list[ip].rv;
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> s{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          s[0] = s[0] + correlation_factor->f12o[io][jo] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op11[jo] / electron_list[jo].wgt;
          s[1] = s[1] + correlation_factor->f12o[io][jo] * (correlation_factor->f13p[ip][jo] - correlation_factor->f23p[ip][jo]) * traces.op12[io][jo] * traces.p23[ip][jo] / electron_list[jo].wgt;
          s[2] = s[2] + correlation_factor->f23p[ip][jo] * traces.op12[io][jo] * traces.ok12[io][jo] / electron_list[jo].wgt;
          s[3] = s[3] + correlation_factor->f23p[ip][jo] * traces.p13[ip][jo]  * traces.ok12[io][jo] / electron_list[jo].wgt;
        }
      }
      t[0] = t[0] + s[0] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list[io].wgt;
      t[1] = t[1] + s[1] * traces.k13[ip][io] / electron_list[io].wgt;
      t[2] = t[2] + s[2] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) / electron_list[io].wgt;
      t[3] = t[3] + s[3] * (correlation_factor->f23p[ip][io] - correlation_factor->f13p[ip][io]) * traces.p23[ip][io] / electron_list[io].wgt;
    }
    direct_1_pair_2_one_ints[4] = direct_1_pair_2_one_ints[4] + t[0] * traces.k12[ip] * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[4] = xchang_1_pair_2_one_ints[4] + t[1] * traces.k12[ip] * electron_pair_list[ip].rv;
    direct_1_pair_2_one_ints[5] = direct_1_pair_2_one_ints[5] + t[2] * traces.p12[ip] * traces.k12[ip] * electron_pair_list[ip].rv;
    xchang_1_pair_2_one_ints[5] = xchang_1_pair_2_one_ints[5] + t[3] * traces.k12[ip] * electron_pair_list[ip].rv;
  }

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    double t_i = 0.0;
    for (int io = 0; io < electron_list->size(); ++io) {
      double t_j = 0.0;
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          std::array<double, 2> t_k{0.0, 0.0};
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
              //         traces.p23[ip][ko] * electron_pair_list[ip].rv / (electron_list[io].wgt * electron_list[jo].wgt * electron_list[ko].wgt);
              //
              t_k[0] = t_k[0] + correlation_factor->f12o[jo][ko] * traces.k13[ip][ko] * traces.p23[ip][ko] / (electron_list[ko].wgt);
              t_k[1] = t_k[1] + correlation_factor->f12o[jo][ko] * traces.v13[ip][ko] * traces.p23[ip][ko] / (electron_list[ko].wgt);
            }
          }
          t_j = t_j + (traces.ok12[io][jo] * t_k[0] - traces.ov12[io][jo] * t_k[1]) * traces.op12[io][jo] / electron_list[jo].wgt;
        }
      }
      t_i = t_i + t_j * (correlation_factor->f13p[ip][io] - correlation_factor->f23p[ip][io]) / electron_list[io].wgt;
    }
    direct_1_pair_3_one_ints[0] = direct_1_pair_3_one_ints[0] + t_i * traces.k12[ip] * electron_pair_list[ip].rv;
  }
}

void MP2F12_VBX_Engine::normalize() {
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

double MP2F12_VBX_Engine::calculate_bx(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  traces.update_bx(electron_pair_list, electron_list);
  zero();
  calculate_bx_t_fa(electron_pair_list, electron_list);
  calculate_bx_t_fb(electron_pair_list, electron_list);
  calculate_bx_t_fc(electron_pair_list, electron_list);
  calculate_bx_t_fd(electron_pair_list, electron_list);
  calculate_bx_k(electron_pair_list, electron_list);
  normalize();

  auto direct_contribution = direct_1_pair_0_one_ints[0] + direct_1_pair_0_one_ints[1]
      + direct_0_pair_2_one_ints[0] + direct_0_pair_2_one_ints[1]
      - 2.0*direct_1_pair_1_one_ints[0] - 2.0*direct_1_pair_1_one_ints[1] + 2.0*direct_1_pair_1_one_ints[2]
      - 2.0*direct_0_pair_3_one_ints[0] -2.0*direct_0_pair_3_one_ints[1]
      + direct_1_pair_2_one_ints[0] + direct_1_pair_2_one_ints[1] - direct_1_pair_2_one_ints[2] - direct_1_pair_2_one_ints[3] - 2.0*direct_1_pair_2_one_ints[4] - 2.0*direct_1_pair_2_one_ints[5]
      + direct_0_pair_4_one_ints[0] - direct_0_pair_4_one_ints[1] + direct_0_pair_4_one_ints[2] - direct_0_pair_4_one_ints[3]
      + 2.0*direct_1_pair_3_one_ints[0];
  auto xchang_contrubtion = xchang_1_pair_0_one_ints[0] + xchang_1_pair_0_one_ints[1] + xchang_0_pair_2_one_ints[0] + xchang_0_pair_2_one_ints[1]
      - 2.0*xchang_1_pair_1_one_ints[0] - 2.0*xchang_1_pair_1_one_ints[1] + 2.0*xchang_1_pair_1_one_ints[2]
      - 2.0*xchang_0_pair_3_one_ints[0] - 2.0*xchang_0_pair_3_one_ints[1]
      + xchang_1_pair_2_one_ints[0] + xchang_1_pair_2_one_ints[1] - xchang_1_pair_2_one_ints[2] - xchang_1_pair_2_one_ints[3] - 2.0*xchang_1_pair_2_one_ints[4] - 2.0*xchang_1_pair_2_one_ints[5]
      + xchang_0_pair_4_one_ints[0] - xchang_0_pair_4_one_ints[1] + xchang_0_pair_4_one_ints[2] - xchang_0_pair_4_one_ints[3]
      + 2.0*xchang_1_pair_3_one_ints[0];
  return c3 * direct_contribution + c4 * xchang_contrubtion;
}

std::pair<double, double> MP2F12_VBX_Engine::calculate_v_vbx(const std::vector<electron_pair_typ>& electron_pair_list, const std::vector<el_one_typ>& electron_list) {
  auto e_v = calculate_v(electron_pair_list, electron_list);
  auto e_bx = calculate_bx(electron_pair_list, electron_list);
  return std::make_pair(e_v, e_v + e_bx);
}
*/
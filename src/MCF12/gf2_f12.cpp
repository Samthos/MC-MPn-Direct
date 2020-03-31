#include <algorithm>
#include <iostream>
#include <numeric>

#include "gf2_f12.h"

GF2_F12_V::GF2_F12_V(IOPs& iops, Basis& basis, std::string extension) :
    MCGF(iops, basis, 0, extension, true),
    traces(basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2, iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]),
    x_traces(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]),
    core_11p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_12p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_22p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_13(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS]),
    core_23(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS])
{
  correlation_factor = create_correlation_factor(iops);
  nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
  nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
}

GF2_F12_V::~GF2_F12_V() {
  delete correlation_factor;
}

void GF2_F12_V::energy_f12(std::vector<std::vector<double>>& egf, std::unordered_map<int, Wavefunction>& wavefunctions, Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  calculate_v(egf, wavefunctions, electron_pair_list, electron_list);
}

double GF2_F12_V::calculate_v_2e(Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    auto f_12 = correlation_factor->f12p[ip];
    core_11p[ip] += c1 * f_12 * traces.p22[ip] * electron_pair_list->rv[ip] * nsamp_pair;
    core_22p[ip] += c1 * f_12 * traces.p11[ip] * electron_pair_list->rv[ip] * nsamp_pair;
    core_12p[ip] += c2 * f_12 * traces.p12[ip] * electron_pair_list->rv[ip] * nsamp_pair;
    core_12p[ip] += c2 * f_12 * traces.p12[ip] * electron_pair_list->rv[ip] * nsamp_pair;
  }
  return 0.0;
}
double GF2_F12_V::calculate_v_3e(Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  double v_1_pair_1_one_int = 0.0;
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      core_13[ip * traces.electrons + io] += -2.0 * nsamp_pair * nsamp_one_1 * c1 * traces.p22[ip] * correlation_factor->f23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_23[ip * traces.electrons + io] += -2.0 * nsamp_pair * nsamp_one_1 * c2 * traces.p12[ip] * correlation_factor->f23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];

      t[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t[2] += correlation_factor->f23[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    core_22p[ip] += -2.0 * nsamp_pair * nsamp_one_1 * c1 * t[1] * electron_pair_list->rv[ip];
    core_12p[ip] += -2.0 * nsamp_pair * nsamp_one_1 * c2 * t[2] * electron_pair_list->rv[ip];
  }
  return 0.0;
}
double GF2_F12_V::calculate_v_4e(Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 8> t{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          t[0] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[3] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[4] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[7] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];

          t[1] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.k13[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[2] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.k13[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[5] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo] * traces.v13[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
          t[6] += correlation_factor->f12o[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * traces.v13[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        }
      }
      core_13[ip * traces.electrons + io] += t[0] * c1 * nsamp_pair * nsamp_one_2 * traces.k13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_13[ip * traces.electrons + io] -= t[4] * c1 * nsamp_pair * nsamp_one_2 * traces.v13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_13[ip * traces.electrons + io] += t[2] * c2 * nsamp_pair * nsamp_one_2 * traces.k23[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_13[ip * traces.electrons + io] -= t[6] * c2 * nsamp_pair * nsamp_one_2 * traces.v23[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];

      core_23[ip * traces.electrons + io] += t[1] * c1 * nsamp_pair * nsamp_one_2 * traces.k23[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_23[ip * traces.electrons + io] -= t[5] * c1 * nsamp_pair * nsamp_one_2 * traces.v23[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_23[ip * traces.electrons + io] += t[3] * c2 * nsamp_pair * nsamp_one_2 * traces.k13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_23[ip * traces.electrons + io] -= t[7] * c2 * nsamp_pair * nsamp_one_2 * traces.v13[ip * traces.electrons + io] * electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
    }
  }
  return 0.0;
}
void GF2_F12_V::calculate_v(std::vector<std::vector<double>>& egf, std::unordered_map<int, Wavefunction>& wavefunctions, Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);

  std::fill(core_11p.begin(), core_11p.end(), 0.0);
  std::fill(core_12p.begin(), core_12p.end(), 0.0);
  std::fill(core_22p.begin(), core_22p.end(), 0.0);
  std::fill(core_13.begin(), core_13.end(), 0.0);
  std::fill(core_23.begin(), core_23.end(), 0.0);

  calculate_v_2e(electron_pair_list, electron_list);
  calculate_v_3e(electron_pair_list, electron_list);
  calculate_v_4e(electron_pair_list, electron_list);
  for (int band = 0; band < numBand; band++) {
    x_traces.set(band, offBand, wavefunctions);
    egf[band][0] += std::inner_product(core_11p.begin(), core_11p.end(), x_traces.x11.begin(), 0.0);
    egf[band][0] += std::inner_product(core_22p.begin(), core_22p.end(), x_traces.x22.begin(), 0.0);
    egf[band][0] += std::inner_product(core_12p.begin(), core_12p.end(), x_traces.x12.begin(), 0.0);
    egf[band][0] += std::inner_product(core_13.begin(), core_13.end(), x_traces.x13.begin(), 0.0);
    egf[band][0] += std::inner_product(core_23.begin(), core_23.end(), x_traces.x23.begin(), 0.0);
  }
}

void GF2_F12_V::core(OVPs& ovps, Electron_Pair_List* electron_pair_list) {}

void GF2_F12_V::energy_no_diff(std::vector<std::vector<double>>&, std::unordered_map<int, Wavefunction>&, Electron_Pair_List*, Tau*) {}

void GF2_F12_V::energy_diff(std::vector<std::vector<double>>&, std::unordered_map<int, Wavefunction>&, Electron_Pair_List*, Tau*) {}

GF2_F12_VBX::GF2_F12_VBX(IOPs& iops, Basis& basis) : GF2_F12_V(iops, basis, "f12_VBX"),
    core_11o(iops.iopns[KEYS::ELECTRONS]),
    core_12o(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    core_d11p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_d12p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_d21p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_d22p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    core_d13(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS]),
    core_d23(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS]),
    T_ip(iops.iopns[KEYS::ELECTRON_PAIRS]),
    T_ip_io(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS]),
    T_ip_jo(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS]),
    T_io_jo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    T_io_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    T_io_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    T_jo_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    T_jo_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]),
    T_ko_lo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS]) 
{
  nsamp_one_3 = nsamp_one_2 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-2);
  nsamp_one_4 = nsamp_one_3 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]-3);
}

void GF2_F12_VBX::energy_f12(std::vector<std::vector<double>>& egf, std::unordered_map<int, Wavefunction>& wavefunctions, Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {
  std::vector<std::vector<double>> e_v(egf.size(), std::vector<double>(egf[0].size(), 0.0));

  calculate_v(e_v, wavefunctions, electron_pair_list, electron_list);
  calculate_bx(egf, wavefunctions, electron_pair_list, electron_list);
  for (int band = 0; band < numBand; band++) {
    egf[band][0] += 2.0 * e_v[band][0];
  }
}

void GF2_F12_VBX::calculate_bx(std::vector<std::vector<double>>& egf, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  traces.update_bx(wavefunctions, electron_pair_list, electron_list);

  std::fill(core_11o.begin(), core_11o.end(), 0.0);
  std::fill(core_12o.begin(), core_12o.end(), 0.0);

  std::fill(core_11p.begin(), core_11p.end(), 0.0);
  std::fill(core_12p.begin(), core_12p.end(), 0.0);
  std::fill(core_22p.begin(), core_22p.end(), 0.0);

  std::fill(core_13.begin(), core_13.end(), 0.0);
  std::fill(core_23.begin(), core_23.end(), 0.0);

  std::fill(core_d11p.begin(), core_d11p.end(), 0.0);
  std::fill(core_d12p.begin(), core_d12p.end(), 0.0);
  std::fill(core_d21p.begin(), core_d21p.end(), 0.0);
  std::fill(core_d22p.begin(), core_d22p.end(), 0.0);

  std::fill(core_d13.begin(), core_d13.end(), 0.0);
  std::fill(core_d23.begin(), core_d23.end(), 0.0);

  calculate_bx_t_fa(electron_pair_list, electron_list);
  calculate_bx_t_fb(electron_pair_list, electron_list);
  calculate_bx_t_fc(electron_pair_list, electron_list);
  calculate_bx_k(electron_pair_list, electron_list);
  for (int band = 0; band < numBand; band++) {
    x_traces.set(band, offBand, wavefunctions);
    x_traces.set_derivative_traces(band, offBand, wavefunctions, electron_pair_list, electron_list);
    egf[band][0] += std::inner_product(core_11o.begin(), core_11o.end(), x_traces.ox11.begin(), 0.0);
    egf[band][0] += std::inner_product(core_12o.begin(), core_12o.end(), x_traces.ox12.begin(), 0.0);

    egf[band][0] += std::inner_product(core_11p.begin(), core_11p.end(), x_traces.x11.begin(), 0.0);
    egf[band][0] += std::inner_product(core_12p.begin(), core_12p.end(), x_traces.x12.begin(), 0.0);
    egf[band][0] += std::inner_product(core_22p.begin(), core_22p.end(), x_traces.x22.begin(), 0.0);

    egf[band][0] += std::inner_product(core_13.begin(), core_13.end(), x_traces.x13.begin(), 0.0);
    egf[band][0] += std::inner_product(core_23.begin(), core_23.end(), x_traces.x23.begin(), 0.0);

    egf[band][0] += std::inner_product(core_d11p.begin(), core_d11p.end(), x_traces.dx11.begin(), 0.0);
    egf[band][0] += std::inner_product(core_d12p.begin(), core_d12p.end(), x_traces.dx12.begin(), 0.0);
    egf[band][0] += std::inner_product(core_d21p.begin(), core_d21p.end(), x_traces.dx21.begin(), 0.0);
    egf[band][0] += std::inner_product(core_d22p.begin(), core_d22p.end(), x_traces.dx22.begin(), 0.0);

    egf[band][0] += std::inner_product(core_d13.begin(), core_d13.end(), x_traces.dx31.begin(), 0.0);
    egf[band][0] += std::inner_product(core_d23.begin(), core_d23.end(), x_traces.dx32.begin(), 0.0);
  }

  if (!correlation_factor->f12_d_is_zero()) {
    traces.update_bx_fd_traces(wavefunctions, electron_list);
    for (int band = 0; band < numBand; band++) {
      x_traces.set(band, offBand, wavefunctions);
      x_traces.set_derivative_traces(band, offBand, wavefunctions, electron_pair_list, electron_list);
      x_traces.set_fd_derivative_traces(band, offBand, wavefunctions, electron_pair_list, electron_list);

      egf[band][0] += calculate_bx_t_fd(electron_pair_list, electron_list);
    }
  }
}

double GF2_F12_VBX::calculate_bx_t_fa_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto cf = correlation_factor->f12p[ip] * correlation_factor->f12p_a[ip] * nsamp_pair;
    core_11p[ip] += c3 * cf * traces.p22[ip] * electron_pair_list->rv[ip];
    core_22p[ip] += c3 * cf * traces.p11[ip] * electron_pair_list->rv[ip];
    core_12p[ip] += c4 * cf * traces.p12[ip] * electron_pair_list->rv[ip];
    core_12p[ip] += c4 * cf * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fa_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      auto f_a = correlation_factor->f12p_a[ip] * correlation_factor->f23[ip * traces.electrons + io];
      auto wgt = -2.0 * electron_pair_list->rv[ip] * electron_list->inverse_weight[io] * nsamp_pair * nsamp_one_1;

      core_13[ip*traces.electrons+io] += f_a * c3  * traces.p22[ip]   * traces.k13[ip * traces.electrons + io] * wgt;
      core_22p[ip] += f_a * c3 * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * wgt;

      core_23[ip*traces.electrons+io] += f_a * c4  * traces.p12[ip]   * traces.k13[ip * traces.electrons + io] * wgt;
      core_12p[ip] += f_a * c4 * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * wgt;
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fa_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      for(int jo = 0; jo < electron_list->size();++jo) {
        if (jo != io) {
          auto f_a = correlation_factor->f12p_a[ip] * correlation_factor->f12o[io * traces.electrons + jo];
          auto wgt = electron_pair_list->rv[ip] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo] * nsamp_pair * nsamp_one_2;
        core_13[ip*traces.electrons+io] += c3 * f_a * traces.k13[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * wgt;
        core_23[ip*traces.electrons+jo] += c3 * f_a * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;

        core_13[ip*traces.electrons+io] -= c3 * f_a * traces.v13[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * wgt;
        core_23[ip*traces.electrons+jo] -= c3 * f_a * traces.p13[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;

        core_13[ip*traces.electrons+jo] += c4 * f_a * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
        core_23[ip*traces.electrons+io] += c4 * f_a * traces.k13[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + jo] * traces.k23[ip * traces.electrons + jo] * wgt;
                                        
        core_13[ip*traces.electrons+jo] -= c4 * f_a * traces.p23[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
        core_23[ip*traces.electrons+io] -= c4 * f_a * traces.v13[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + jo] * traces.v23[ip * traces.electrons + jo] * wgt;
        }
      }
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fa(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fa_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_4e(electron_pair_list, electron_list);
  return en;
}

double GF2_F12_VBX::calculate_bx_t_fb_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        auto wgt = 2.0 * nsamp_one_2 * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
        auto f_b = correlation_factor->f12o_b[io * traces.electrons + jo] * correlation_factor->f12o[io * traces.electrons + jo];
        core_11o[io] += c3 * f_b * traces.op11[jo] * wgt;
        core_12o[io * traces.electrons + jo] += c4 * f_b * traces.op12[io * traces.electrons + jo] * wgt;
      }
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fb_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            auto wgt = -2.0 * nsamp_one_3 * electron_list->inverse_weight[io] *electron_list->inverse_weight[jo] *electron_list->inverse_weight[ko];
            auto cf = correlation_factor->f12o[jo * traces.electrons + ko] * correlation_factor->f12o_b[io * traces.electrons + jo];
          core_12o[io * traces.electrons + ko] += c3 * cf * traces.op11[jo] * traces.ok12[io * traces.electrons + ko] * wgt;
          core_11o[jo]                         += c3 * cf * traces.op12[io * traces.electrons + ko]   * traces.ok12[io * traces.electrons + ko] * wgt;
          core_12o[jo * traces.electrons + ko] += c4 * cf * traces.op12[io * traces.electrons + jo] * traces.ok12[io * traces.electrons + ko] * wgt;
          core_12o[io * traces.electrons + jo] += c4 * cf * traces.op12[jo * traces.electrons + ko]   * traces.ok12[io * traces.electrons + ko] * wgt;
          }
        }
      }
    }
  }
  return (t[0] + t[1]);
}
void calculate_bx_t_fb_4e_help(
    std::vector<double>& T_ko_lo, const std::vector<double>& S_ko_lo,
    std::vector<double>& T_jo_lo, const std::vector<double>& S_jo_lo_1, const std::vector<double>& S_jo_lo_2,
    std::vector<double>& T_io_lo, const std::vector<double>& S_io_jo,
    std::vector<double>& T_io_ko,
    std::vector<double>& out, const std::vector<double>& S_io_ko,
    const std::vector<double>& weight,
    double alpha, size_t size
    ) {
  for (int ko = 0; ko < size; ++ko) {
    for (int lo = 0; lo < size; ++lo) {
      T_ko_lo[ko * size + lo] = S_ko_lo[ko * size + lo] * weight[ko] * weight[lo];
    }
  }

  std::transform(S_jo_lo_1.begin(), S_jo_lo_1.end(), S_jo_lo_2.begin(), T_jo_lo.begin(), std::multiplies<>());

  std::fill(T_io_lo.begin(), T_io_lo.end(), 0.0);
  for (int io = 0; io < size; ++io) {
    for (int jo = 0; jo < size; ++jo) {
      for (int lo = 0; lo < size; ++lo) {
        T_io_lo[io*size+lo] += S_io_jo[io * size + jo] * T_jo_lo[jo * size + lo];
      }
    }
  }
  for (int io = 0; io < size; ++io) {
    T_io_lo[io*size+io] = 0.0;
  }

  std::fill(T_io_ko.begin(), T_io_ko.end(), 0.0);
  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      for (int lo = 0; lo < size; ++lo) {
        T_io_ko[io * size + ko] += T_ko_lo[ko * size + lo] * T_io_lo[io*size+lo];
      }
    }
  }

  std::transform(T_ko_lo.begin(), T_ko_lo.end(), T_jo_lo.begin(), T_ko_lo.begin(), std::multiplies<>());
  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      for (int lo = 0; lo < size; ++lo) {
        T_io_ko[io * size + ko] -= T_ko_lo[ko * size + lo] * S_io_jo[io * size + ko];
      }
    }
  }

  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      T_io_ko[io * size + ko] += T_ko_lo[io * size + ko] * S_io_jo[io * size + ko];
    }
  }

  for (int io = 0; io < size; ++io) {
    for (int ko = 0; ko < size; ++ko) {
      out[io * size + ko] += alpha * T_io_ko[io * size + ko] * S_io_ko[io * size + ko];
    }
  }
}
double GF2_F12_VBX::calculate_bx_t_fb_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  double c_c3 = 2.0 * c3 * nsamp_one_4;
  double c_c4 = 2.0 * c4 * nsamp_one_4;

  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      T_io_jo[io * traces.electrons + jo] = correlation_factor->f12o_b[io * traces.electrons + jo] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
    }
  }

  calculate_bx_t_fb_4e_help(
      T_ko_lo, correlation_factor->f12o,
      T_jo_lo, traces.op12, traces.ok12,
      T_io_lo, T_io_jo,
      T_io_ko,
      core_12o, traces.ok12,
      electron_list->inverse_weight, c_c3, electron_list->size());

  calculate_bx_t_fb_4e_help(
      T_ko_lo, correlation_factor->f12o,
      T_jo_lo, traces.op12, traces.ov12,
      T_io_lo, T_io_jo,
      T_io_ko,
      core_12o, traces.ov12,
      electron_list->inverse_weight, -c_c3, electron_list->size());

  std::copy(T_io_jo.begin(), T_io_jo.end(), T_io_lo.begin());
  for (int jo = 0; jo < electron_list->size(); ++jo) {
    for (int ko = 0; ko < electron_list->size(); ++ko) {
      T_jo_ko[jo * traces.electrons + ko] = correlation_factor->f12o[jo * traces.electrons + ko] * electron_list->inverse_weight[jo] * electron_list->inverse_weight[ko];
    }
  }
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      for (int lo = 0; lo < electron_list->size(); ++lo) {
        T_io_jo[                   lo] = T_io_lo[io * traces.electrons + lo] * traces.ok12[jo * traces.electrons + lo];
        T_io_jo[traces.electrons + lo] = T_io_lo[io * traces.electrons + lo] * traces.ov12[jo * traces.electrons + lo];
      }
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        std::array<double, 2> t_ko{0.0, 0.0};
        for (int lo = 0; lo < electron_list->size(); ++lo) {
          t_ko[0] += T_io_jo[                   lo] * traces.op12[ko * traces.electrons + lo];
          t_ko[1] += T_io_jo[traces.electrons + lo] * traces.op12[ko * traces.electrons + lo];
        }
        core_12o[io * traces.electrons + jo] += t_ko[0] *c_c4 * T_jo_ko[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko];
        core_12o[io * traces.electrons + jo] -= t_ko[1] *c_c4 * T_jo_ko[jo * traces.electrons + ko] * traces.ov12[io * traces.electrons + ko];
      }
    }
  }
  for (int io = 0; io < traces.electrons; io++) {
    core_12o[io*traces.electrons+io] = 0.0;
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fb(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fb_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fb_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fb_4e(electron_pair_list, electron_list);
  return en;
}

double GF2_F12_VBX::calculate_bx_t_fc_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    auto cf = correlation_factor->f12p_c[ip] * correlation_factor->f12p[ip] * nsamp_pair;
    core_11p[ip]  += c3 * cf * traces.dp22[ip] * electron_pair_list->rv[ip];
    core_d22p[ip] += c3 * cf * traces.p11[ip] * electron_pair_list->rv[ip];
    core_d12p[ip] += c4 * cf * traces.p12[ip] * electron_pair_list->rv[ip];
    core_12p[ip]  += c4 * cf * traces.dp12[ip] * electron_pair_list->rv[ip];
                      
    core_d11p[ip] -= c3 * cf * traces.p22[ip] * electron_pair_list->rv[ip];
    core_22p[ip]  -= c3 * cf * traces.dp11[ip] * electron_pair_list->rv[ip];
    core_12p[ip]  -= c4 * cf * traces.dp21[ip] * electron_pair_list->rv[ip];
    core_d21p[ip] -= c4 * cf * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fc_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      auto cf = -2.0 * correlation_factor->f23[ip * traces.electrons + io] * correlation_factor->f12p_c[ip] * nsamp_pair * nsamp_one_1;
      auto wgt = electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      core_13[ip*traces.electrons+io]  += c3 * cf * traces.dp22[ip] * traces.k13[ip * traces.electrons + io] * wgt;
      core_d23[ip*traces.electrons+io] += c4 * cf * traces.p12[ip] * traces.k13[ip * traces.electrons + io] * wgt;
      core_d13[ip*traces.electrons+io] -= c3 * cf * traces.p22[ip]  * traces.k13[ip * traces.electrons + io] * wgt;
      core_23[ip*traces.electrons+io]  -= c4 * cf * traces.dp21[ip] * traces.k13[ip * traces.electrons + io] * wgt;

      core_d22p[ip]    += c3 * cf * traces.p13[ip * traces.electrons + io]  * traces.k13[ip * traces.electrons + io] * wgt;
      core_12p[ip]     += c4 * cf * traces.dp32[ip][io] * traces.k13[ip * traces.electrons + io] * wgt;
      core_22p[ip]     -= c3 * cf * traces.dp31[ip][io]  * traces.k13[ip * traces.electrons + io] * wgt;
      core_d21p[ip]    -= c4 * cf * traces.p23[ip * traces.electrons + io]  * traces.k13[ip * traces.electrons + io] * wgt;
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fc_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          auto cf = correlation_factor->f12p_c[ip] * correlation_factor->f12o[io * traces.electrons + jo];
          auto wgt = electron_pair_list->rv[ip] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo] * nsamp_pair * nsamp_one_2;
core_d23[ip*traces.electrons+jo] += c3 * cf * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_d23[ip*traces.electrons+jo] -= c3 * cf * traces.p13[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
core_d23[ip*traces.electrons+io] += c4 * cf * traces.p13[ip * traces.electrons + jo] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_d23[ip*traces.electrons+io] -= c4 * cf * traces.p13[ip * traces.electrons + jo] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;

core_d13[ip*traces.electrons+io] -= c3 * cf * traces.p23[ip * traces.electrons + jo] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_d13[ip*traces.electrons+io] += c3 * cf * traces.p23[ip * traces.electrons + jo] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
core_d13[ip*traces.electrons+jo] -= c4 * cf * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_d13[ip*traces.electrons+jo] += c4 * cf * traces.p23[ip * traces.electrons + io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;

core_23[ip*traces.electrons+jo] -= c3 * cf * traces.dp31[ip][io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_23[ip*traces.electrons+jo] += c3 * cf * traces.dp31[ip][io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
core_23[ip*traces.electrons+io] -= c4 * cf * traces.dp31[ip][jo] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_23[ip*traces.electrons+io] += c4 * cf * traces.dp31[ip][jo] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;

core_13[ip*traces.electrons+io] += c3 * cf * traces.dp32[ip][jo] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_13[ip*traces.electrons+io] -= c3 * cf * traces.dp32[ip][jo] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
core_13[ip*traces.electrons+jo] += c4 * cf * traces.dp32[ip][io] * traces.k13[ip * traces.electrons + io] * traces.k23[ip * traces.electrons + jo] * wgt;
core_13[ip*traces.electrons+jo] -= c4 * cf * traces.dp32[ip][io] * traces.v13[ip * traces.electrons + io] * traces.v23[ip * traces.electrons + jo] * wgt;
        }
      }
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_t_fc(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fc_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fc_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fc_4e(electron_pair_list, electron_list);
  return en;
}

double GF2_F12_VBX::calculate_bx_t_fd_2e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        auto f_d = correlation_factor->f12o_d[io * traces.electrons + jo] * correlation_factor->f12o[io * traces.electrons + jo];
        auto wgt = 1.0 * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
        t[0] += f_d * (x_traces.ox11[io] * traces.ds_p22[io][jo] - x_traces.ds_x11[io][jo] * traces.op11[jo]) * wgt;
        t[0] += f_d * (traces.op11[io] * x_traces.ds_x22[io][jo] - traces.ds_p11[io][jo] * x_traces.ox11[jo]) * wgt;
        t[1] += f_d * (x_traces.ox12[io*traces.electrons+jo] * traces.ds_p12[io][jo] - x_traces.ox12[io*traces.electrons+jo] * traces.ds_p21[io][jo]) * wgt;
        t[1] += f_d * (traces.op12[io * traces.electrons + jo] * x_traces.ds_x12[io][jo] - traces.op12[io * traces.electrons + jo] * x_traces.ds_x21[io][jo]) * wgt;
      }
    }
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_2;
}
double GF2_F12_VBX::calculate_bx_t_fd_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            auto wgt = 1.0 * electron_list->inverse_weight[io]* electron_list->inverse_weight[jo] * electron_list->inverse_weight[ko];
            auto cf = correlation_factor->f12o[jo * traces.electrons + ko] * correlation_factor->f12o_d[io * traces.electrons + jo];
      t[0] += cf * traces.ok12[io * traces.electrons + ko] * (x_traces.ox12[io*traces.electrons+ko] * traces.ds_p22[io][jo]   - x_traces.ds_x31[io][jo][ko] * traces.op11[jo]      ) * wgt;
      t[0] += cf * traces.ok12[io * traces.electrons + ko] * (traces.op12[io * traces.electrons + ko] * x_traces.ds_x22[io][jo]   - traces.ds_p31[io][jo][ko] * x_traces.ox11[jo]      ) * wgt;

      t[1] += cf * traces.ok12[io * traces.electrons + ko] * (x_traces.ds_x32[io][jo][ko] * traces.op12[io * traces.electrons + jo]     -  x_traces.ox12[jo*traces.electrons+ko]      * traces.ds_p21[io][jo]) * wgt;
      t[1] += cf * traces.ok12[io * traces.electrons + ko] * (traces.ds_p32[io][jo][ko] * x_traces.ox12[io*traces.electrons+jo]     -  traces.op12[jo * traces.electrons + ko]      * x_traces.ds_x21[io][jo]) * wgt;
          }
        }
      }
    }
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_one_3;
}
double GF2_F12_VBX::calculate_bx_t_fd_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      if (jo != io) {
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          if (ko != jo && ko != io) {
            for (int lo = 0; lo < electron_list->size(); ++lo) {
              if (lo != ko && lo != jo && lo != io) {
                auto cf = correlation_factor->f12o[ko * traces.electrons + lo] * correlation_factor->f12o_d[io * traces.electrons + jo]; 
                auto wgt = 1.0 * electron_list->inverse_weight[io]* electron_list->inverse_weight[jo] * electron_list->inverse_weight[ko] * electron_list->inverse_weight[lo];
   t[0] += cf * (x_traces.ds_x32[io][jo][lo] * traces.op12[io * traces.electrons + ko] - x_traces.ox12[jo*traces.electrons+lo] * traces.ds_p31[io][jo][ko]) * traces.ok12[io * traces.electrons + ko] * traces.ok12[jo * traces.electrons + lo] * wgt;
   t[0] += cf * (traces.ds_p32[io][jo][lo] * x_traces.ox12[io*traces.electrons+ko] - traces.op12[jo * traces.electrons + lo] * x_traces.ds_x31[io][jo][ko]) * traces.ok12[io * traces.electrons + ko] * traces.ok12[jo * traces.electrons + lo] * wgt;
   t[0] -= cf * (x_traces.ds_x32[io][jo][lo] * traces.op12[io * traces.electrons + ko] - x_traces.ox12[jo*traces.electrons+lo] * traces.ds_p31[io][jo][ko]) * traces.ov12[io * traces.electrons + ko] * traces.ov12[jo * traces.electrons + lo] * wgt;
   t[0] -= cf * (traces.ds_p32[io][jo][lo] * x_traces.ox12[io*traces.electrons+ko] - traces.op12[jo * traces.electrons + lo] * x_traces.ds_x31[io][jo][ko]) * traces.ov12[io * traces.electrons + ko] * traces.ov12[jo * traces.electrons + lo] * wgt;

   t[1] += cf * (x_traces.ox12[io*traces.electrons+lo] * traces.ds_p32[io][jo][ko] - x_traces.ds_x31[io][jo][lo] * traces.op12[jo * traces.electrons + ko]) * traces.ok12[io * traces.electrons + ko] * traces.ok12[jo * traces.electrons + lo] * wgt;
   t[1] += cf * (traces.op12[io * traces.electrons + lo] * x_traces.ds_x32[io][jo][ko] - traces.ds_p31[io][jo][lo] * x_traces.ox12[jo*traces.electrons+ko]) * traces.ok12[io * traces.electrons + ko] * traces.ok12[jo * traces.electrons + lo] * wgt;
   t[1] -= cf * (x_traces.ox12[io*traces.electrons+lo] * traces.ds_p32[io][jo][ko] - x_traces.ds_x31[io][jo][lo] * traces.op12[jo * traces.electrons + ko]) * traces.ov12[io * traces.electrons + ko] * traces.ov12[jo * traces.electrons + lo] * wgt;
   t[1] -= cf * (traces.op12[io * traces.electrons + lo] * x_traces.ds_x32[io][jo][ko] - traces.ds_p31[io][jo][lo] * x_traces.ox12[jo*traces.electrons+ko]) * traces.ov12[io * traces.electrons + ko] * traces.ov12[jo * traces.electrons + lo] * wgt;
              }
            }
          }
        }
      }
    }
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_4;
}
double GF2_F12_VBX::calculate_bx_t_fd(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fd_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fd_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fd_4e(electron_pair_list, electron_list);
  return en;
}

double GF2_F12_VBX::calculate_bx_k_3e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  double c = 2.0 * nsamp_pair * nsamp_one_1;
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      auto wgt = electron_pair_list->rv[ip] * electron_list->inverse_weight[io];
      auto cf = correlation_factor->f23[ip * traces.electrons + io] * (correlation_factor->f23[ip * traces.electrons + io] - correlation_factor->f13[ip * traces.electrons + io]);
      core_12p[ip] += c * c3 * cf * traces.op11[io] * traces.k12[ip] * wgt;
      core_11o[io] += c * c3 * cf * traces.p12[ip]  * traces.k12[ip] * wgt;

      core_13[ip*traces.electrons+io] += c * c4 * cf * traces.p23[ip * traces.electrons + io] * traces.k12[ip] * wgt;
      core_23[ip*traces.electrons+io] += c * c4 * cf * traces.p13[ip * traces.electrons + io] * traces.k12[ip] * wgt;
    }
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_k_4e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};

  double c = -2.0 * nsamp_pair * nsamp_one_2;
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        if (jo != io) {
          auto f34o = correlation_factor->f12o[io * traces.electrons + jo];

          auto f14p = correlation_factor->f13[ip * traces.electrons + jo];
          auto f24p = correlation_factor->f23[ip * traces.electrons + jo];

          auto f23 = correlation_factor->f23[ip * traces.electrons + io];
          auto f13 = correlation_factor->f13[ip * traces.electrons + io];
          auto wgt = electron_pair_list->rv[ip] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];

core_12p[ip] += c * c3 * f24p * (f23 - f13) * traces.op12[io * traces.electrons + jo]      * traces.ok12[io * traces.electrons + jo] * traces.k12[ip] * wgt;
core_11o[jo] += c * c3 * f34o * (f14p - f24p) * traces.p23[ip * traces.electrons + io]  * traces.k13[ip * traces.electrons + io]  * traces.k12[ip] * wgt;

core_12o[io*traces.electrons+jo] += c * c4 * f34o * (f14p - f24p) * traces.p23[ip * traces.electrons + jo]  * traces.k13[ip * traces.electrons + io]  * traces.k12[ip] * wgt;
core_12o[io*traces.electrons+jo] += c * c3 * f24p * (f23 - f13) * traces.p12[ip]      * traces.ok12[io * traces.electrons + jo] * traces.k12[ip] * wgt;

core_13[ip*traces.electrons+jo] += c * c4 * f24p * (f23 - f13)   * traces.p23[ip * traces.electrons + io]  * traces.ok12[io * traces.electrons + jo] * traces.k12[ip] * wgt;
core_23[ip*traces.electrons+jo] += c * c4 * f34o * (f14p - f24p) * traces.op12[io * traces.electrons + jo]  * traces.k13[ip * traces.electrons + io]  * traces.k12[ip] * wgt;
core_23[ip*traces.electrons+io] += c * c3 * f34o * (f14p - f24p) * traces.op11[jo]     * traces.k13[ip * traces.electrons + io]  * traces.k12[ip] * wgt;
core_23[ip*traces.electrons+io] += c * c4 * f24p * (f23 - f13)   * traces.p13[ip * traces.electrons + jo]  * traces.ok12[io * traces.electrons + jo] * traces.k12[ip] * wgt;
        }
      }
    }
  }
  return (t[0] + t[1]);
}
void calculate_bx_k_5e_help_1(
    std::vector<double>& T_jo_ko,
    std::vector<double>& T_io_jo, const std::vector<double>& S_io_ko,
    std::vector<double>& T_ip_io, const std::vector<double>& S_ip_jo,
    std::vector<double>& out, const std::vector<double> S_ip_io,
    double alpha, size_t size, size_t size_ep
    ) {
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      size, size, size,
      1.0,
      T_jo_ko.data(), size,
      S_io_ko.data(), size,
      0.0,
      T_io_jo.data(), size);
  cblas_dscal(size, 0.0, T_io_jo.data(), size+1);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      size_ep, size, size,
      1.0,
      T_io_jo.data(), size,
      S_ip_jo.data(), size,
      0.0,
      T_ip_io.data(), size);

  for (int ip = 0, idx=0; ip < size_ep; ip++) {
    for (int io = 0; io < size; ++io, ++idx) {
      out[idx] += alpha * T_ip_io[idx] * S_ip_io[idx];
    }
  }
}
void calculate_bx_k_5e_help_2(
    std::vector<double>& T_io_jo, const std::vector<double>& S_ip_jo, const std::vector<double>& S_ip_io,
    std::vector<double>& T_jo_ko, const std::vector<double>& S_io_ko,
    std::vector<double>& out, const std::vector<double> S_jo_ko,
    double alpha, size_t size, size_t size_ep
    ) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      size, size, size_ep,
      1.0,
      S_ip_jo.data(), size,
      S_ip_io.data(), size,
      0.0,
      T_io_jo.data(), size);
  cblas_dscal(size, 0.0, T_io_jo.data(), size+1);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      size, size, size,
      1.0,
      T_io_jo.data(), size,
      S_io_ko.data(), size,
      0.0,
      T_jo_ko.data(), size);
  cblas_dscal(size, 0.0, T_io_jo.data(), size+1);

  for (int jo = 0, idx=0; jo < size; jo++) {
    for (int ko = 0; ko < size; ++ko, ++idx) {
      out[idx] += alpha * T_jo_ko[idx] * S_jo_ko[idx];
    }
  }
}
double GF2_F12_VBX::calculate_bx_k_5e(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  double c_c3 = 2.0 * nsamp_pair * nsamp_one_3 * c3;
  double c_c4 = 2.0 * nsamp_pair * nsamp_one_3 * c4;

  std::transform(traces.k12.begin(), traces.k12.end(), electron_pair_list->rv.begin(), T_ip.begin(), std::multiplies<>());

  std::transform(correlation_factor->f13.begin(), correlation_factor->f13.end(), correlation_factor->f23.begin(), T_ip_jo.begin(), std::minus<>());
  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      T_ip_jo[ip*traces.electrons+jo] *= T_ip[ip] * electron_list->inverse_weight[jo];
    }
  }

  for (int io = 0; io < electron_list->size(); ++io) {
    for (int ko = 0; ko < electron_list->size(); ++ko) {
      T_io_ko[io * traces.electrons + ko] = correlation_factor->f12o[io * traces.electrons + ko] * electron_list->inverse_weight[io] * electron_list->inverse_weight[ko];
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  size_t size = electron_list->size();
  size_t size_ep = electron_pair_list->size();

  std::transform(traces.op12.begin(), traces.op12.end(), traces.ok12.begin(), T_jo_ko.begin(), std::multiplies<>());
  calculate_bx_k_5e_help_1(
    T_jo_ko, 
    T_io_jo, T_io_ko,
    T_ip_io, T_ip_jo,
    core_23, traces.k13,
    c_c3, traces.electrons, traces.electron_pairs);

  std::transform(traces.op12.begin(), traces.op12.end(), traces.ov12.begin(), T_jo_ko.begin(), std::multiplies<>());
  calculate_bx_k_5e_help_1(
    T_jo_ko, 
    T_io_jo, T_io_ko,
    T_ip_io, T_ip_jo,
    core_23, traces.v13,
    -c_c3, traces.electrons, traces.electron_pairs);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::transform(traces.p23.begin(), traces.p23.end(), traces.k13.begin(), T_ip_io.begin(), std::multiplies<>());
  calculate_bx_k_5e_help_2(
    T_io_jo, T_ip_jo, T_ip_io,
    T_jo_ko, T_io_ko,
    core_12o, traces.ok12,
    c_c3, size, size_ep);

  std::transform(traces.p23.begin(), traces.p23.end(), traces.v13.begin(), T_ip_io.begin(), std::multiplies<>());
  calculate_bx_k_5e_help_2(
    T_io_jo, T_ip_jo, T_ip_io,
    T_jo_ko, T_io_ko,
    core_12o, traces.ov12,
    -c_c3, size, size_ep);

  for (int ip = 0; ip < electron_pair_list->size();ip++) {
    for (int io = 0; io < electron_list->size(); ++io) {
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        T_io_jo[0 * traces.electrons + ko] = T_io_ko[io*traces.electrons+ko] * traces.k13[ip * traces.electrons + ko];
        T_io_jo[1 * traces.electrons + ko] = T_io_ko[io*traces.electrons+ko] * traces.v13[ip * traces.electrons + ko];
        T_io_jo[2 * traces.electrons + ko] = T_io_ko[io*traces.electrons+ko] * traces.p23[ip * traces.electrons + ko];
        T_io_jo[3 * traces.electrons + ko] = T_io_ko[io*traces.electrons+ko] * traces.p23[ip * traces.electrons + ko];
      }
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        std::array<double, 4> t_jo{0.0, 0.0, 0.0, 0.0};
        for (int ko = 0; ko < electron_list->size(); ++ko) {
          t_jo[0] += T_io_jo[0 * traces.electrons + ko] * traces.op12[jo * traces.electrons + ko];
          t_jo[1] += T_io_jo[1 * traces.electrons + ko] * traces.op12[jo * traces.electrons + ko];
          t_jo[2] += T_io_jo[2 * traces.electrons + ko] * traces.ok12[jo * traces.electrons + ko];
          t_jo[3] += T_io_jo[3 * traces.electrons + ko] * traces.ov12[jo * traces.electrons + ko];
        }
        core_23[ip*traces.electrons+io]  += t_jo[0] * c_c4 * T_ip_jo[ip * traces.electrons + jo] * traces.ok12[io * traces.electrons + jo];
        core_23[ip*traces.electrons+io]  -= t_jo[1] * c_c4 * T_ip_jo[ip * traces.electrons + jo] * traces.ov12[io * traces.electrons + jo];
                                                      
        core_12o[io*traces.electrons+jo] += t_jo[2] * c_c4 * T_ip_jo[ip * traces.electrons + jo] * traces.k13[ip * traces.electrons + io];
        core_12o[io*traces.electrons+jo] -= t_jo[3] * c_c4 * T_ip_jo[ip * traces.electrons + jo] * traces.v13[ip * traces.electrons + io];
      }
    }
  }
  for (int io = 0; io < electron_list->size(); io++) {
    core_12o[io * traces.electrons + io] = 0.0;
  }
  return (t[0] + t[1]);
}
double GF2_F12_VBX::calculate_bx_k(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  double en = 0.0;
  en += calculate_bx_k_3e(electron_pair_list, electron_list);
  en += calculate_bx_k_4e(electron_pair_list, electron_list);
  en += calculate_bx_k_5e(electron_pair_list, electron_list);
  return en;
}

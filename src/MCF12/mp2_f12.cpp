#include <algorithm>
#include <iostream>
#include <numeric>

#include "cblas.h"
#include "mp2_f12.h"

double calculate_v_4e_help(
    std::vector<double>& T_ip_io, const std::vector<double>& S_ip_jo, const std::vector<double>& S_io_jo,
    const std::vector<double>& S_ip_io_1, const std::vector<double>& S_ip_io_2,
    const std::vector<double>& S_ip, size_t size, size_t size_ep
    ) {
  cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans,
      size, size_ep, size,
      1.0,
      S_io_jo.data(), size,
      S_ip_jo.data(), size,
      0.0,
      T_ip_io.data(), size);

  double en = 0.0;
  for (int ip = 0, idx = 0; ip < size_ep; ip++) {
    double t = 0.0;
    for (int io = 0; io < size; ++io, ++idx) {
      t += T_ip_io[idx] * S_ip_io_1[idx] * S_ip_io_2[idx];
      }
    en += t * S_ip[ip];
  }
  return en;
}

MP2_F12_V::MP2_F12_V(const IOPs& iops, std::string extension) :
    F12_MP_Functional(0, 0, extension),
    traces(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]),
    T_ip_io(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    T_ip_jo(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    T_io_jo(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    correlation_factor(new Correlation_Factor_Data(iops))
{
  nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
  nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
}

MP2_F12_V::~MP2_F12_V() {
}

void MP2_F12_V::energy(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  calculate_v(emp, control, wavefunctions, electron_pair_list, electron_list);
}

double MP2_F12_V::calculate_v_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    auto f_12 = correlation_factor->f12p[ip];
    t[0] += f_12 * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    t[1] += f_12 * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return (c1 * t[0] + c2 * t[1]) * nsamp_pair;
}
double MP2_F12_V::calculate_v_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    std::array<double, 2> t_ip{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t_ip[0] += correlation_factor->f23[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    t[0] += t_ip[0] * traces.p22[ip] * electron_pair_list->rv[ip];
    t[1] += t_ip[1] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return -2.0 * (c1 * t[0] + c2 * t[1]) * nsamp_pair * nsamp_one_1;
}
double MP2_F12_V::calculate_v_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  for (int io = 0, idx = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo, ++idx) {
      T_io_jo[idx] = correlation_factor->f12o[idx] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
    }
  }
  std::array<double, 2> t{0.0, 0.0};
  std::transform(traces.p23.begin(), traces.p23.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.k13, electron_pair_list->rv, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p23.begin(), traces.p23.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.v13, electron_pair_list->rv, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p13.begin(), traces.p13.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.k13, electron_pair_list->rv, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p13.begin(), traces.p13.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.v13, electron_pair_list->rv, electron_list->size(), electron_pair_list->size());
  return (c1 * t[0] + c2 * t[1]) * nsamp_pair * nsamp_one_2;
}
void MP2_F12_V::calculate_v(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);
  emp += calculate_v_2e(electron_pair_list, electron_list);
  emp += calculate_v_3e(electron_pair_list, electron_list);
  emp += calculate_v_4e(electron_pair_list, electron_list);
}

MP2_F12_VBX::MP2_F12_VBX(const IOPs& iops) : MP2_F12_V(iops, "f12_VBX"),
  T_ip(iops.iopns[KEYS::ELECTRON_PAIRS], 0.0),
  T_jo_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
  T_io_ko(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0)
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
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    auto f_a = correlation_factor->f12p_a[ip];
    auto f_12 = correlation_factor->f12p[ip];
    t[0] += f_12 * (f_a + 2.0 * c1 / c3) * traces.p11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
    t[1] += f_12 * (f_a + 2.0 * c2 / c4) * traces.p12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_pair;
}
double MP2_F12_VBX::calculate_bx_t_fa_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 2> t_ip{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t_ip[0] += correlation_factor->f23[ip * traces.electrons + io] * traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_a = correlation_factor->f12p_a[ip];
    t[0] += t_ip[0] * (f_a + 2.0 * c1 / c3) * traces.p22[ip] * electron_pair_list->rv[ip];
    t[1] += t_ip[1] * (f_a + 2.0 * c2 / c4) * traces.p12[ip] * electron_pair_list->rv[ip];
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_1;
}
double MP2_F12_VBX::calculate_bx_t_fa_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0, idx = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo, ++idx) {
      T_io_jo[idx] = correlation_factor->f12o[idx] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
    }
  }

  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    T_ip[ip] = (correlation_factor->f12p_a[ip] + 2.0 * c1 / c3) * electron_pair_list->rv[ip];
  }
  std::transform(traces.p23.begin(), traces.p23.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p23.begin(), traces.p23.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());

  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    T_ip[ip] = (correlation_factor->f12p_a[ip] + 2.0 * c2 / c4) * electron_pair_list->rv[ip];
  }
  std::transform(traces.p13.begin(), traces.p13.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p13.begin(), traces.p13.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());
  return (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fa(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_t_fa_2e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_3e(electron_pair_list, electron_list);
  en += calculate_bx_t_fa_4e(electron_pair_list, electron_list);
  return en;
}

double MP2_F12_VBX::calculate_bx_t_fb_2e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      auto f_b = correlation_factor->f12o_b[io * traces.electrons + jo];
      t_io[0] += f_b * correlation_factor->f12o[io * traces.electrons + jo] * traces.op11[jo] * electron_list->inverse_weight[jo];
      t_io[1] += f_b * correlation_factor->f12o[io * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
    }
    t[0] += t_io[0] * traces.op11[io] * electron_list->inverse_weight[io];
    t[1] += t_io[1] * electron_list->inverse_weight[io];
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_one_2;
}
double MP2_F12_VBX::calculate_bx_t_fb_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
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
    t[0] += t_io[0] * electron_list->inverse_weight[io];
    t[1] += t_io[1] * electron_list->inverse_weight[io];
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_one_3;
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
double MP2_F12_VBX::calculate_bx_t_fb_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  t[0] += calculate_bx_t_fb_4e_help(traces.ok12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);
  t[0] -= calculate_bx_t_fb_4e_help(traces.ov12, traces.op12, correlation_factor->f12o_b, correlation_factor->f12o, electron_list->inverse_weight, traces.electrons);

  for (int jo = 0; jo < electron_list->size(); ++jo) {
    for (int ko = 0; ko < electron_list->size(); ++ko) {
      T_jo_ko[jo * traces.electrons + ko] = traces.op12[jo * traces.electrons + ko] * electron_list->inverse_weight[ko] * electron_list->inverse_weight[jo];
    }
  }
  for (int io = 0; io < electron_list->size(); ++io) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      for (int lo = 0; lo < electron_list->size(); ++lo) {
        T_io_jo[jo * traces.electrons + lo] = T_jo_ko[io * traces.electrons + lo] * traces.ok12[jo * traces.electrons + lo];
      }
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        traces.electrons, traces.electrons, traces.electrons,
        1.0,
        correlation_factor->f12o.data(), traces.electrons, // ko lo
        T_io_jo.data(), traces.electrons, // jo lo 
        0.0,
        T_io_ko.data(), traces.electrons); // jo ko 
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      double t_jo = 0.0;
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        t_jo += T_io_ko[jo * traces.electrons + ko] * T_jo_ko[jo * traces.electrons + ko] * traces.ok12[io * traces.electrons + ko];
      }
      t_io[0] += t_jo * correlation_factor->f12o_b[io * traces.electrons + jo];
    }
            
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      for (int lo = 0; lo < electron_list->size(); ++lo) {
        T_io_jo[jo * traces.electrons + lo] = T_jo_ko[io * traces.electrons + lo] * traces.ov12[jo * traces.electrons + lo];
      }
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        traces.electrons, traces.electrons, traces.electrons,
        1.0,
        correlation_factor->f12o.data(), traces.electrons, // ko lo
        T_io_jo.data(), traces.electrons, // jo lo 
        0.0,
        T_io_ko.data(), traces.electrons); // jo ko 
    for (int jo = 0; jo < electron_list->size(); ++jo) {
      double t_jo = 0.0;
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        t_jo += T_io_ko[jo * traces.electrons + ko] * T_jo_ko[jo * traces.electrons + ko] * traces.ov12[io * traces.electrons + ko];
      }
      t_io[1] += t_jo * correlation_factor->f12o_b[io * traces.electrons + jo];
    }
    t[1] += t_io[0];
    t[1] -= t_io[1];
  }
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
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    auto f_12 = correlation_factor->f12p[ip];
    auto f_c = correlation_factor->f12p_c[ip];
    t[0] += f_12 * f_c * (traces.p11[ip] * traces.dp22[ip] - traces.dp11[ip] * traces.p22[ip]) * electron_pair_list->rv[ip];
    t[1] += f_12 * f_c * (traces.dp12[ip] * traces.p12[ip] - traces.p12[ip] * traces.dp21[ip]) * electron_pair_list->rv[ip];
  }
  return (c3 * t[0] + c4 * t[1]) * nsamp_pair;
}
double MP2_F12_VBX::calculate_bx_t_fc_3e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  for(int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t_ip{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t_ip[0] += correlation_factor->f23[ip * traces.electrons + io] *  traces.p13[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[1] += correlation_factor->f23[ip * traces.electrons + io] * traces.dp31[ip*traces.electrons+io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];

      t_ip[2] += correlation_factor->f23[ip * traces.electrons + io] * traces.dp32[ip*traces.electrons+io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[3] += correlation_factor->f23[ip * traces.electrons + io] *  traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    auto f_c = correlation_factor->f12p_c[ip];
    t[0] += (t_ip[0] * traces.dp22[ip] - t_ip[1] *  traces.p22[ip]) * f_c * electron_pair_list->rv[ip];
    t[1] += (t_ip[2] *  traces.p12[ip] - t_ip[3] * traces.dp21[ip]) * f_c * electron_pair_list->rv[ip];
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_1;
}
double MP2_F12_VBX::calculate_bx_t_fc_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  for (int io = 0, idx = 0; io < electron_list->size(); ++io) {
    for (int jo = 0; jo < electron_list->size(); ++jo, ++idx) {
      T_io_jo[idx] = correlation_factor->f12o[idx] * electron_list->inverse_weight[io] * electron_list->inverse_weight[jo];
    }
  }
  std::transform(correlation_factor->f12p_c.begin(), correlation_factor->f12p_c.end(), electron_pair_list->rv.begin(), T_ip.begin(), std::multiplies<>());

  std::array<double, 2> t{0.0, 0.0};

  std::transform(traces.dp32.begin(), traces.dp32.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p23.begin(), traces.p23.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.dp31, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.dp32.begin(), traces.dp32.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p13, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p23.begin(), traces.p23.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[0] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.dp31, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p13.begin(), traces.p13.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.dp32, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.dp31.begin(), traces.dp31.end(), traces.k23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.k13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.p13.begin(), traces.p13.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] -= calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.dp32, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());

  std::transform(traces.dp31.begin(), traces.dp31.end(), traces.v23.begin(), T_ip_jo.begin(), std::multiplies<>());
  t[1] += calculate_v_4e_help(T_ip_io, T_ip_jo, T_io_jo, traces.p23, traces.v13, T_ip, electron_list->size(), electron_pair_list->size());

  return (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_2;
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
      t_io[0] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op11[io] * traces.ds_p22[io][jo] - traces.ds_p11[io][jo] * traces.op11[jo]) * electron_list->inverse_weight[jo];
      t_io[1] += f_d * correlation_factor->f12o[io * traces.electrons + jo] * (traces.op12[io * traces.electrons + jo] * traces.ds_p12[io][jo] - traces.ds_p21[io][jo] * traces.op12[io * traces.electrons + jo]) * electron_list->inverse_weight[jo];
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
  std::array<double, 2> t{0.0, 0.0};
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 2> t_io{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      t_io[0] += correlation_factor->f23[ip * traces.electrons + io] * (correlation_factor->f23[ip * traces.electrons + io] - correlation_factor->f13[ip * traces.electrons + io]) * traces.op11[io] * electron_list->inverse_weight[io];
      t_io[1] += correlation_factor->f23[ip * traces.electrons + io] * (correlation_factor->f23[ip * traces.electrons + io] - correlation_factor->f13[ip * traces.electrons + io]) * traces.p13[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    t[0] += t_io[0] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    t[1] += t_io[1] * traces.k12[ip] * electron_pair_list->rv[ip];
  }
  return 2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_1;
}
double MP2_F12_VBX::calculate_bx_k_4e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  std::transform(correlation_factor->f13.begin(), correlation_factor->f13.end(), correlation_factor->f23.begin(), T_ip_jo.begin(), std::minus<>());
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    std::array<double, 4> t_ip{0.0, 0.0, 0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      std::array<double, 4> t_io{0.0, 0.0, 0.0, 0.0};
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        t_io[0] += correlation_factor->f12o[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op11[jo] * electron_list->inverse_weight[jo];
        t_io[1] += correlation_factor->f12o[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.p23[ip * traces.electrons + jo] * electron_list->inverse_weight[jo];
        t_io[2] += correlation_factor->f23[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo] * traces.ok12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
        t_io[3] += correlation_factor->f23[ip * traces.electrons + jo] * traces.p13[ip * traces.electrons + jo]  * traces.ok12[io * traces.electrons + jo] * electron_list->inverse_weight[jo];
      }
      t_ip[0] += t_io[0] * traces.p23[ip * traces.electrons + io] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[1] += t_io[1] * traces.k13[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[2] -= t_io[2] * T_ip_jo[ip * traces.electrons + io] * electron_list->inverse_weight[io];
      t_ip[3] -= t_io[3] * T_ip_jo[ip * traces.electrons + io] * traces.p23[ip * traces.electrons + io] * electron_list->inverse_weight[io];
    }
    t[0] += t_ip[0] * traces.k12[ip] * electron_pair_list->rv[ip];
    t[1] += t_ip[1] * traces.k12[ip] * electron_pair_list->rv[ip];
    t[0] += t_ip[2] * traces.p12[ip] * traces.k12[ip] * electron_pair_list->rv[ip];
    t[1] += t_ip[3] * traces.k12[ip] * electron_pair_list->rv[ip];
  }
  return -2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_2;
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
      size, size, size,
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
double MP2_F12_VBX::calculate_bx_k_5e(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  std::array<double, 2> t{0.0, 0.0};
  std::transform(electron_pair_list->rv.begin(), electron_pair_list->rv.end(), traces.k12.begin(), T_ip.begin(), std::multiplies<>());
  for (int ip = 0, idx = 0; ip < electron_pair_list->size(); ++ip) {
    for (int jo = 0; jo < electron_list->size(); ++jo, ++idx) {
      T_ip_jo[idx] = (correlation_factor->f13[ip * traces.electrons + jo] - correlation_factor->f23[ip * traces.electrons + jo]) * T_ip[ip];
    }
  }

  t[0] += calculate_bx_k_5e_help(traces.k13, traces.p23, traces.ok12,traces.op12, correlation_factor->f12o, T_ip_jo, 
      electron_list->inverse_weight, electron_list->size(), electron_pair_list->size());
  t[0] -= calculate_bx_k_5e_help(traces.v13, traces.p23, traces.ov12,traces.op12, correlation_factor->f12o, T_ip_jo, 
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
    std::array<double, 2> t_ip{0.0, 0.0};
    for (int io = 0; io < electron_list->size(); ++io) {
      for (int ko = 0; ko < electron_list->size(); ++ko) {
        T_io_jo[io * traces.electrons + ko] = T_io_ko[io * traces.electrons + ko] * traces.p23[ip * traces.electrons + ko];
      }
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        traces.electrons, traces.electrons, traces.electrons,
        1.0,
        traces.ok12.data(), traces.electrons, // jo ko
        T_io_jo.data(), traces.electrons, // io ko 
        0.0,
        T_jo_ko.data(), traces.electrons); // io jo 
    for (int io = 0; io < electron_list->size(); ++io) {
      double t_io = 0.0;
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        t_io += T_jo_ko[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo]; 
      }
      t_ip[0] += t_io * traces.k13[ip * traces.electrons + io];
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        traces.electrons, traces.electrons, traces.electrons,
        1.0,
        traces.ov12.data(), traces.electrons, // jo ko
        T_io_jo.data(), traces.electrons, // io ko 
        0.0,
        T_jo_ko.data(), traces.electrons); // io jo 
    for (int io = 0; io < electron_list->size(); ++io) {
      double t_io = 0.0;
      for (int jo = 0; jo < electron_list->size(); ++jo) {
        t_io += T_jo_ko[io * traces.electrons + jo] * T_ip_jo[ip * traces.electrons + jo] * traces.op12[io * traces.electrons + jo]; 
      }
      t_ip[0] -= t_io * traces.v13[ip * traces.electrons + io];
    }

    t[1] += t_ip[0] * traces.k12[ip];
  }
  return 2.0 * (c3 * t[0] + c4 * t[1]) * nsamp_pair * nsamp_one_3;
}
double MP2_F12_VBX::calculate_bx_k(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  double en = 0.0;
  en += calculate_bx_k_3e(electron_pair_list, electron_list);
  en += calculate_bx_k_4e(electron_pair_list, electron_list);
  en += calculate_bx_k_5e(electron_pair_list, electron_list);
  return en;
}

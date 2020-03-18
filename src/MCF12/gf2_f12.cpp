#include <algorithm>
#include <iostream>
#include <numeric>

#include "gf2_f12.h"

GF2_F12_V::GF2_F12_V(IOPs& iops, Basis& basis) :
    MCGF(iops, basis, 0, "f12_V", true),
    traces(basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2, iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS])
{
  correlation_factor = create_correlation_factor(iops);
  nsamp_pair = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp_one_1 = 1.0 / static_cast<double>(iops.iopns[KEYS::ELECTRONS]);
  nsamp_one_2 = nsamp_one_1 / static_cast<double>(iops.iopns[KEYS::ELECTRONS] - 1.0);
}
GF2_F12_V::~GF2_F12_V() {
  delete correlation_factor;
}

void GF2_F12_V::core(OVPs& ovps, Electron_Pair_List* electron_pair_list) {}
void GF2_F12_V::energy_no_diff(std::vector<std::vector<double>>&, 
   std::unordered_map<int, Wavefunction>&,
   Electron_Pair_List*, Tau*) {}
void GF2_F12_V::energy_diff(std::vector<std::vector<double>>&,
   std::unordered_map<int, Wavefunction>&,
   Electron_Pair_List*, Tau*) {}

void GF2_F12_V::energy_f12(std::vector<std::vector<double>>& egf, 
   std::unordered_map<int, Wavefunction>& wavefunctions,
   Electron_Pair_List* electron_pair_list, Electron_List* electron_list) {

  const double* psi_ep_1;
  const double* psi_ep_2;
  const double* psi;
  size_t lda = wavefunctions[WC::electrons].lda;

  std::vector<double> x11(electron_pair_list->size());
  std::vector<double> x12(electron_pair_list->size());
  std::vector<double> x22(electron_pair_list->size());
  std::vector<std::vector<double>> x13(electron_pair_list->size(), std::vector<double>(electron_list->size()));
  std::vector<std::vector<double>> x23(electron_pair_list->size(), std::vector<double>(electron_list->size()));

  traces.update_v(wavefunctions);
  correlation_factor->update(electron_pair_list, electron_list);
  for (int band = 0; band < numBand; band++) {

    psi_ep_1 = wavefunctions[WC::electron_pairs_1].vir() + (band-offBand);
    psi_ep_2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);
    psi = wavefunctions[WC::electrons].vir() + (band-offBand);

    for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
      x11[ip] = psi_ep_1[ip * lda] * psi_ep_1[ip * lda];
      x12[ip] = psi_ep_1[ip * lda] * psi_ep_2[ip * lda];
      x22[ip] = psi_ep_2[ip * lda] * psi_ep_2[ip * lda];
      for (int io = 0; io < electron_list->size(); ++io) {
        x13[ip][io] = psi_ep_1[ip * lda] * psi[io * lda];
        x23[ip][io] = psi_ep_2[ip * lda] * psi[io * lda];
      }
    }

    double v_1_pair_0_one_int = 0.0;
    for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
      auto f_12 = correlation_factor->calculate_f12(electron_pair_list->r12[ip]);
       v_1_pair_0_one_int += c1 * f_12 * x11[ip] * traces.p22[ip] * electron_pair_list->rv[ip];
       v_1_pair_0_one_int += c1 * f_12 * traces.p11[ip] * x22[ip] * electron_pair_list->rv[ip];
       v_1_pair_0_one_int += c2 * f_12 * x12[ip] * traces.p12[ip] * electron_pair_list->rv[ip];
       v_1_pair_0_one_int += c2 * f_12 * traces.p12[ip] * x12[ip] * electron_pair_list->rv[ip];
    }
    v_1_pair_0_one_int *= nsamp_pair;

    double v_1_pair_1_one_int = 0.0;
    for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
      std::array<double, 4> t{0.0, 0.0, 0.0, 0.0};
      for (int io = 0; io < electron_list->size(); ++io) {
        t[0] += correlation_factor->f23p[ip][io] * x13[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
        t[1] += correlation_factor->f23p[ip][io] * traces.p13[ip][io] * traces.k13[ip][io] / electron_list->weight[io];

        t[2] += correlation_factor->f23p[ip][io] * traces.p23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
        t[3] += correlation_factor->f23p[ip][io] * x23[ip][io] * traces.k13[ip][io] / electron_list->weight[io];
      }
      v_1_pair_1_one_int += c1 * t[0] * traces.p22[ip] * electron_pair_list->rv[ip];
      v_1_pair_1_one_int += c1 * t[1] * x22[ip] * electron_pair_list->rv[ip];
      
      v_1_pair_1_one_int += c2 * t[2] * x12[ip] * electron_pair_list->rv[ip];
      v_1_pair_1_one_int += c2 * t[3] * traces.p12[ip] * electron_pair_list->rv[ip];
    }
    v_1_pair_1_one_int *= -2.0 * nsamp_pair * nsamp_one_1;

    std::array<double, 2> v_1_pair_2_one_int{0.0, 0.0};
    for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
      for (int io = 0; io < electron_list->size(); ++io) {
        std::array<double, 8> t{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int jo = 0; jo < electron_list->size(); ++jo) {
          if (jo != io) {
            t[0] += correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];
            t[1] += correlation_factor->f12o[io][jo] * x23[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];

            t[2] += correlation_factor->f12o[io][jo] * x13[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];
            t[3] += correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.k23[ip][jo] / electron_list->weight[jo];

            t[4] += correlation_factor->f12o[io][jo] * traces.p23[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
            t[5] += correlation_factor->f12o[io][jo] * x23[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
               
            t[6] += correlation_factor->f12o[io][jo] * x13[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
            t[7] += correlation_factor->f12o[io][jo] * traces.p13[ip][jo] * traces.v23[ip][jo] / electron_list->weight[jo];
          }
        }
        v_1_pair_2_one_int[0] += t[0] * x13[ip][io] * traces.k13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
        v_1_pair_2_one_int[0] += t[1] * traces.p13[ip][io] * traces.k13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
                                     
        v_1_pair_2_one_int[1] += t[2] * traces.p23[ip][io] * traces.k13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
        v_1_pair_2_one_int[1] += t[3] * x23[ip][io] * traces.k13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
                                     
        v_1_pair_2_one_int[0] -= t[4] * x13[ip][io] * traces.v13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
        v_1_pair_2_one_int[0] -= t[5] * traces.p13[ip][io] * traces.v13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
                                    
        v_1_pair_2_one_int[1] -= t[6] * traces.p23[ip][io] * traces.v13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
        v_1_pair_2_one_int[1] -= t[7] * x23[ip][io] * traces.v13[ip][io] * electron_pair_list->rv[ip] / electron_list->weight[io];
      }
    }
    v_1_pair_2_one_int[0] *= c1 * nsamp_pair * nsamp_one_2;
    v_1_pair_2_one_int[1] *= c2 * nsamp_pair * nsamp_one_2;

    egf[band][0] += v_1_pair_0_one_int + v_1_pair_2_one_int[0] + v_1_pair_2_one_int[1] + v_1_pair_1_one_int;
  }
}


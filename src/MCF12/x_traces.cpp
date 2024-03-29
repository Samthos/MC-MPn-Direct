#include <algorithm>

#include "x_traces.h"

X_Traces::X_Traces(int n_e_p, int n_e) :
    n_electron_pairs(n_e_p),
    n_electrons(n_e),
    x11(n_electron_pairs),
    x12(n_electron_pairs),
    x22(n_electron_pairs),
    dx11(n_electron_pairs),
    dx12(n_electron_pairs),
    dx21(n_electron_pairs),
    dx22(n_electron_pairs),
    x13(n_electron_pairs * n_electrons),
    x23(n_electron_pairs * n_electrons),
    dx31(n_electron_pairs * n_electrons),
    dx32(n_electron_pairs * n_electrons),
    ox11(n_electrons),
    ox12(n_electrons * n_electrons),
    ds_x11(n_electrons, std::vector<double>(n_electrons)),
    ds_x22(n_electrons, std::vector<double>(n_electrons)),
    ds_x12(n_electrons, std::vector<double>(n_electrons)),
    ds_x21(n_electrons, std::vector<double>(n_electrons)),
    ds_x31(n_electrons, std::vector<std::vector<double>>(n_electrons, std::vector<double>(n_electrons, 0.0))),
    ds_x32(n_electrons, std::vector<std::vector<double>>(n_electrons, std::vector<double>(n_electrons, 0.0)))
{}


void X_Traces::set(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions) {
  const double* psi_ep_1 = wavefunctions[WC::electron_pairs_1].vir() + band - offBand;
  const double* psi_ep_2 = wavefunctions[WC::electron_pairs_2].vir() + band - offBand;
  const double* psi = wavefunctions[WC::electrons].vir() + band - offBand;
  size_t lda = wavefunctions[WC::electrons].lda;

  for (int ip = 0, idx = 0; ip < n_electron_pairs; ++ip) {
    x11[ip] = psi_ep_1[ip * lda] * psi_ep_1[ip * lda];
    x12[ip] = psi_ep_1[ip * lda] * psi_ep_2[ip * lda];
    x22[ip] = psi_ep_2[ip * lda] * psi_ep_2[ip * lda];
    for (int io = 0; io < n_electrons; ++io, idx++) {
      x13[idx] = psi_ep_1[ip * lda] * psi[io * lda];
      x23[idx] = psi_ep_2[ip * lda] * psi[io * lda];
    }
  }
  for (int io = 0, idx = 0; io < n_electrons; ++io) {
    ox11[io] = psi[io * lda] * psi[io * lda];
    for (int jo = 0; jo < n_electrons; ++jo, ++idx) {
      ox12[idx] = psi[io * lda] * psi[jo * lda];
    }
    ox12[io*n_electrons+io] = 0.0;
  }
}

void X_Traces::set_derivative_traces(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  const double* psi_ep_1 = wavefunctions[WC::electron_pairs_1].vir() + band - offBand;
  const double* psi_ep_2 = wavefunctions[WC::electron_pairs_2].vir() + band - offBand;
  const double* psi = wavefunctions[WC::electrons].vir() + band - offBand;

  const double* psi_ep_1_dx = wavefunctions[WC::electron_pairs_1_dx].vir() + band - offBand;
  const double* psi_ep_2_dx = wavefunctions[WC::electron_pairs_2_dx].vir() + band - offBand;
  const double* psi_dx = wavefunctions[WC::electrons_dx].vir() + band - offBand;

  const double* psi_ep_1_dy = wavefunctions[WC::electron_pairs_1_dy].vir() + band - offBand;
  const double* psi_ep_2_dy = wavefunctions[WC::electron_pairs_2_dy].vir() + band - offBand;
  const double* psi_dy = wavefunctions[WC::electrons_dy].vir() + band - offBand;

  const double* psi_ep_1_dz = wavefunctions[WC::electron_pairs_1_dz].vir() + band - offBand;
  const double* psi_ep_2_dz = wavefunctions[WC::electron_pairs_2_dz].vir() + band - offBand;
  const double* psi_dz = wavefunctions[WC::electrons_dz].vir() + band - offBand;

  size_t lda = wavefunctions[WC::electrons].lda;

  for (int ip = 0, idx = 0; ip < n_electron_pairs; ip++) {
    auto dr = electron_pair_list->pos1[ip] - electron_pair_list->pos2[ip];
    dx11[ip] = psi_ep_1[ip * lda] * (dr[0] * psi_ep_1_dx[ip * lda] + dr[1] * psi_ep_1_dy[ip * lda] + dr[2] * psi_ep_1_dz[ip * lda]);
    dx12[ip] = psi_ep_1[ip * lda] * (dr[0] * psi_ep_2_dx[ip * lda] + dr[1] * psi_ep_2_dy[ip * lda] + dr[2] * psi_ep_2_dz[ip * lda]);
    dx21[ip] = psi_ep_2[ip * lda] * (dr[0] * psi_ep_1_dx[ip * lda] + dr[1] * psi_ep_1_dy[ip * lda] + dr[2] * psi_ep_1_dz[ip * lda]);
    dx22[ip] = psi_ep_2[ip * lda] * (dr[0] * psi_ep_2_dx[ip * lda] + dr[1] * psi_ep_2_dy[ip * lda] + dr[2] * psi_ep_2_dz[ip * lda]);
    for (int io = 0; io < n_electrons; ++io, idx++) {
      dx31[idx] = psi[io * lda] * (dr[0] * psi_ep_1_dx[ip * lda] + dr[1] * psi_ep_1_dy[ip * lda] + dr[2] * psi_ep_1_dz[ip * lda]);
      dx32[idx] = psi[io * lda] * (dr[0] * psi_ep_2_dx[ip * lda] + dr[1] * psi_ep_2_dy[ip * lda] + dr[2] * psi_ep_2_dz[ip * lda]);
    }
  }
}

void X_Traces::set_fd_derivative_traces(int band, int offBand, std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  const double* psi = wavefunctions[WC::electrons].vir() + band - offBand;
  const double* psi_dx = wavefunctions[WC::electrons_dx].vir() + band - offBand;
  const double* psi_dy = wavefunctions[WC::electrons_dy].vir() + band - offBand;
  const double* psi_dz = wavefunctions[WC::electrons_dz].vir() + band - offBand;

  size_t lda = wavefunctions[WC::electrons].lda;
  for (int io = 0; io < n_electrons; ++io) {
    for (int jo = 0; jo < n_electrons; ++jo) {
      auto dr = electron_list->pos[io] - electron_list->pos[jo];
      ds_x11[io][jo] = psi[io * lda] * (dr[0] * psi_dx[io * lda] + dr[1] * psi_dy[io * lda] + dr[2] * psi_dz[io * lda]);
      ds_x12[io][jo] = psi[io * lda] * (dr[0] * psi_dx[jo * lda] + dr[1] * psi_dy[jo * lda] + dr[2] * psi_dz[jo * lda]);
      ds_x21[io][jo] = psi[jo * lda] * (dr[0] * psi_dx[io * lda] + dr[1] * psi_dy[io * lda] + dr[2] * psi_dz[io * lda]);
      ds_x22[io][jo] = psi[jo * lda] * (dr[0] * psi_dx[jo * lda] + dr[1] * psi_dy[jo * lda] + dr[2] * psi_dz[jo * lda]);
      for (int ko = 0; ko < n_electrons; ++ko) {
        ds_x31[io][jo][ko] = psi[ko * lda] * (dr[0] * psi_dx[io * lda] + dr[1] * psi_dy[io * lda] + dr[2] * psi_dz[io * lda]);
        ds_x32[io][jo][ko] = psi[ko * lda] * (dr[0] * psi_dx[jo * lda] + dr[1] * psi_dy[jo * lda] + dr[2] * psi_dz[jo * lda]);
      }
    }
  }
}

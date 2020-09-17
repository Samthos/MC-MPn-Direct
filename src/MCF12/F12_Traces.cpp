//
// Created by aedoran on 1/2/20.
//

#include <algorithm>
#include <numeric>
#include <array>

#include "cblas.h"
#include "blas_calls.h"

#include "F12_Traces.h"

F12_Traces::F12_Traces(int electron_pairs_, int electrons_) :
    electron_pairs(electron_pairs_),
    electrons(electrons_),
    op11(electrons, 0.0),
    op12(electrons * electrons, 0.0),
    ok12(electrons * electrons, 0.0),
    ov12(electrons * electrons, 0.0),
    p11(electron_pairs, 0.0),
    p12(electron_pairs, 0.0),
    p22(electron_pairs, 0.0),
    k12(electron_pairs, 0.0),
    dp11(electron_pairs, 0.0),
    dp12(electron_pairs, 0.0),
    dp21(electron_pairs, 0.0),
    dp22(electron_pairs, 0.0),
    p13(electron_pairs * electrons, 0.0),
    k13(electron_pairs * electrons, 0.0),
    v13(electron_pairs * electrons, 0.0),
    dp31(electron_pairs * electrons, 0.0),
    p23(electron_pairs * electrons, 0.0),
    k23(electron_pairs * electrons, 0.0),
    v23(electron_pairs * electrons, 0.0),
    dp32(electron_pairs * electrons, 0.0),
    ds_p11(electrons, std::vector<double>(electrons, 0.0)),
    ds_p12(electrons, std::vector<double>(electrons, 0.0)),
    ds_p21(electrons, std::vector<double>(electrons, 0.0)),
    ds_p22(electrons, std::vector<double>(electrons, 0.0)),
    ds_p31(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0))),
    ds_p32(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0)))
{
}

void F12_Traces::update_v(std::unordered_map<int, Wavefunction_Type>& wavefunctions) {
  build_one_e_one_e_traces(wavefunctions[WC::electrons]);
  build_two_e_traces(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2]);
  build_two_e_one_e_traces(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], wavefunctions[WC::electrons]);
}

void F12_Traces::update_bx(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  build_two_e_derivative_traces(wavefunctions, electron_pair_list);
  build_two_e_one_e_derivative_traces(wavefunctions, electron_pair_list, electron_list);
}

void F12_Traces::update_bx_fd_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_List_Type* electron_list) {
  auto lda = wavefunctions[WC::electrons].lda;
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;
  const double* psi = wavefunctions[WC::electrons].data();
  const double* psi_dx = wavefunctions[WC::electrons_dx].data();
  const double* psi_dy = wavefunctions[WC::electrons_dy].data();
  const double* psi_dz = wavefunctions[WC::electrons_dz].data();

  for (int io = 0; io < electrons; ++io) {
    for (int jo = 0; jo < electrons; ++jo) {
      auto dr = electron_list->pos[io] - electron_list->pos[jo];
      ds_p11[io][jo] = 0.0;
      ds_p12[io][jo] = 0.0;
      ds_p21[io][jo] = 0.0;
      ds_p22[io][jo] = 0.0;
      for (int im = iocc1; im < iocc2; ++im) {
        ds_p11[io][jo] = ds_p11[io][jo] + psi[io * lda + im] * (dr[0] * psi_dx[io * lda + im] + dr[1] * psi_dy[io * lda + im] + dr[2] * psi_dz[io * lda + im]);
        ds_p12[io][jo] = ds_p12[io][jo] + psi[io * lda + im] * (dr[0] * psi_dx[jo * lda + im] + dr[1] * psi_dy[jo * lda + im] + dr[2] * psi_dz[jo * lda + im]);
        ds_p21[io][jo] = ds_p21[io][jo] + psi[jo * lda + im] * (dr[0] * psi_dx[io * lda + im] + dr[1] * psi_dy[io * lda + im] + dr[2] * psi_dz[io * lda + im]);
        ds_p22[io][jo] = ds_p22[io][jo] + psi[jo * lda + im] * (dr[0] * psi_dx[jo * lda + im] + dr[1] * psi_dy[jo * lda + im] + dr[2] * psi_dz[jo * lda + im]);
      }


      for (int ko = 0; ko < electrons; ++ko) {
        ds_p31[io][jo][ko] = 0.0;
        ds_p32[io][jo][ko] = 0.0;
        for (int im = iocc1; im < iocc2; ++im) {
          ds_p31[io][jo][ko] = ds_p31[io][jo][ko] + psi[ko * lda + im] * (dr[0] * psi_dx[io * lda + im] + dr[1] * psi_dy[io * lda + im] + dr[2] * psi_dz[io * lda + im]);
          ds_p32[io][jo][ko] = ds_p32[io][jo][ko] + psi[ko * lda + im] * (dr[0] * psi_dx[jo * lda + im] + dr[1] * psi_dy[jo * lda + im] + dr[2] * psi_dz[jo * lda + im]);
        }
      }
    }
    ds_p11[io][io] = 0.0;
    ds_p12[io][io] = 0.0;
    ds_p21[io][io] = 0.0;
    ds_p22[io][io] = 0.0;
  }
  for (int io = 0; io < electrons; ++io) {
    for (int jo = 0; jo < electrons; ++jo) {
      ds_p31[io][io][jo] = 0.0;
      ds_p31[io][jo][io] = 0.0;
      ds_p31[jo][io][io] = 0.0;

      ds_p32[io][io][jo] = 0.0;
      ds_p32[io][jo][io] = 0.0;
      ds_p32[jo][io][io] = 0.0;
    }
    ds_p31[io][io][io] = 0.0;
    ds_p32[io][io][io] = 0.0;
  }
}

void F12_Traces::build_one_e_one_e_traces(const Wavefunction_Type& electron_psi) {
  double alpha = 1.0;
  double beta = 0.0;
  auto iocc1 = electron_psi.iocc1;
  auto iocc2 = electron_psi.iocc2;
  auto ivir1 = electron_psi.ivir1;
  auto ivir2 = electron_psi.ivir2;
  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      electrons, iocc2 - iocc1,
      alpha,
      electron_psi.data() + iocc1, electron_psi.lda,
      beta,
      op12.data(), electrons);
  set_Upper_from_Lower(op12.data(), electrons);

  for (int io = 0; io < electrons; io++) {
    op11[io] = op12[io * electrons + io];
  }

  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      electrons, ivir2 - ivir1,
      alpha,
      electron_psi.data() + ivir1, electron_psi.lda,
      beta,
      ov12.data(), electrons);
  set_Upper_from_Lower(ov12.data(), electrons);

  beta = 1.0;
  std::copy(op12.begin(), op12.end(), ok12.begin());
  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      electrons, iocc1,
      alpha,
      electron_psi.data(), electron_psi.lda,
      beta,
      ok12.data(), electrons);
  set_Upper_from_Lower(ok12.data(), electrons);

  cblas_dscal(electrons, 0.0, op12.data(), electrons+1);
  cblas_dscal(electrons, 0.0, ov12.data(), electrons+1);
  cblas_dscal(electrons, 0.0, ok12.data(), electrons+1);
}

void F12_Traces::build_two_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2) {
  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  for(int ip = 0; ip < electron_pairs;ip++) {
    p11[ip] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
        electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc2,
        electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
        0.0);
    p12[ip] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
        electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc2,
        electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc1,
        0.0);
    p22[ip] = std::inner_product(electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc1,
        electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc2,
        electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc1,
        0.0);
    k12[ip] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda,
        electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
        electron_pair_psi2.data() + ip * electron_pair_psi2.lda,
        p12[ip]);
  }
}

void F12_Traces::build_two_e_one_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2, const Wavefunction_Type& electron_psi) {
  double alpha = 1.0;
  double beta = 0.0;
  auto iocc1 = electron_psi.iocc1;
  auto iocc2 = electron_psi.iocc2;
  auto ivir1 = electron_psi.ivir1;
  auto ivir2 = electron_psi.ivir2;

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, iocc2 - iocc1,
      alpha,
      electron_psi.data() + iocc1, electron_psi.lda,
      electron_pair_psi1.data() + iocc1, electron_pair_psi2.lda,
      beta,
      p13.data(), electrons);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, iocc2 - iocc1,
      alpha,
      electron_psi.data() + iocc1, electron_psi.lda,
      electron_pair_psi2.data() + iocc1, electron_pair_psi2.lda,
      beta,
      p23.data(), electrons);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, ivir2 - ivir1,
      alpha,
      electron_psi.data() + ivir1, electron_psi.lda,
      electron_pair_psi1.data() + ivir1, electron_pair_psi2.lda,
      beta,
      v13.data(), electrons);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, ivir2 - ivir1,
      alpha,
      electron_psi.data() + ivir1, electron_psi.lda,
      electron_pair_psi2.data() + ivir1, electron_pair_psi2.lda,
      beta,
      v23.data(), electrons);

  beta = 1.0;
  std::copy(p13.begin(), p13.end(), k13.begin());
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, iocc1,
      alpha,
      electron_psi.data(), electron_psi.lda,
      electron_pair_psi1.data(), electron_pair_psi2.lda,
      beta,
      k13.data(), electrons);

  std::copy(p23.begin(), p23.end(), k23.begin());
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      electrons, electron_pairs, iocc1,
      alpha,
      electron_psi.data(), electron_psi.lda,
      electron_pair_psi2.data(), electron_pair_psi2.lda,
      beta,
      k23.data(), electrons);
}

void F12_Traces::build_two_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list) {
  auto lda = wavefunctions[WC::electron_pairs_1].lda;
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;

  const double* psi1 = wavefunctions[WC::electron_pairs_1].data();
  const double* psi2 = wavefunctions[WC::electron_pairs_2].data();

  const double* psi1_dx = wavefunctions[WC::electron_pairs_1_dx].data();
  const double* psi1_dy = wavefunctions[WC::electron_pairs_1_dy].data();
  const double* psi1_dz = wavefunctions[WC::electron_pairs_1_dz].data();

  const double* psi2_dx = wavefunctions[WC::electron_pairs_2_dx].data();
  const double* psi2_dy = wavefunctions[WC::electron_pairs_2_dy].data();
  const double* psi2_dz = wavefunctions[WC::electron_pairs_2_dz].data();

  for (int ip = 0; ip < electron_pairs; ip++) {
    auto dr = electron_pair_list->pos1[ip] - electron_pair_list->pos2[ip];
    dp11[ip] = 0.0;
    dp12[ip] = 0.0;
    dp21[ip] = 0.0;
    dp22[ip] = 0.0;
    for (int im = iocc1, idx = ip * lda + iocc1; im < iocc2; ++im, ++idx) {
      dp11[ip] = dp11[ip] + psi1[idx] * (dr[0] * psi1_dx[idx] + dr[1] * psi1_dy[idx] + dr[2] * psi1_dz[idx]);
      dp12[ip] = dp12[ip] + psi1[idx] * (dr[0] * psi2_dx[idx] + dr[1] * psi2_dy[idx] + dr[2] * psi2_dz[idx]);
      dp21[ip] = dp21[ip] + psi2[idx] * (dr[0] * psi1_dx[idx] + dr[1] * psi1_dy[idx] + dr[2] * psi1_dz[idx]);
      dp22[ip] = dp22[ip] + psi2[idx] * (dr[0] * psi2_dx[idx] + dr[1] * psi2_dy[idx] + dr[2] * psi2_dz[idx]);
    }
  }
}

void F12_Traces::build_two_e_one_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;
  auto o_lda = wavefunctions[WC::electrons].lda;
  auto p_lda = wavefunctions[WC::electron_pairs_1].lda;

  const double* psi = wavefunctions[WC::electrons].data();

  const double* psi1_dx = wavefunctions[WC::electron_pairs_1_dx].data();
  const double* psi1_dy = wavefunctions[WC::electron_pairs_1_dy].data();
  const double* psi1_dz = wavefunctions[WC::electron_pairs_1_dz].data();

  const double* psi2_dx = wavefunctions[WC::electron_pairs_2_dx].data();
  const double* psi2_dy = wavefunctions[WC::electron_pairs_2_dy].data();
  const double* psi2_dz = wavefunctions[WC::electron_pairs_2_dz].data();

  for(int ip = 0, idx=0; ip < electron_pairs; ++ip) {
    auto dr = electron_pair_list->pos1[ip] - electron_pair_list->pos2[ip];
    for(int io = 0; io < electrons; ++io, ++idx) {
      dp31[idx] = 0.0;
      dp32[idx] = 0.0;
      for(int im = iocc1, p_idx = ip * p_lda + iocc1, o_idx = io * o_lda + iocc1; im < iocc2; ++im, p_idx++, o_idx++) {
        dp31[idx] += psi[o_idx] * (dr[0] * psi1_dx[p_idx] + dr[1] * psi1_dy[p_idx] + dr[2] * psi1_dz[p_idx]);
        dp32[idx] += psi[o_idx] * (dr[0] * psi2_dx[p_idx] + dr[1] * psi2_dy[p_idx] + dr[2] * psi2_dz[p_idx]);
      }
    }
  }
}

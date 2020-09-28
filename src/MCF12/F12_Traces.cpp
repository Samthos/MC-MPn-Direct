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
    ds_p11(electrons * electrons, 0.0),
    ds_p12(electrons * electrons, 0.0),
    ds_p21(electrons * electrons, 0.0),
    ds_p22(electrons * electrons, 0.0),
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
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;

  const vector_double& psi1_dx = wavefunctions[WC::electron_pairs_1_dx].psi;
  const vector_double& psi1_dy = wavefunctions[WC::electron_pairs_1_dy].psi;
  const vector_double& psi1_dz = wavefunctions[WC::electron_pairs_1_dz].psi;

  const vector_double& psi2_dx = wavefunctions[WC::electron_pairs_2_dx].psi;
  const vector_double& psi2_dy = wavefunctions[WC::electron_pairs_2_dy].psi;
  const vector_double& psi2_dz = wavefunctions[WC::electron_pairs_2_dz].psi;

  vector_double& psi1_tau_dx= wavefunctions[WC::electron_pairs_1_dx].psiTau;
  vector_double& psi1_tau_dy= wavefunctions[WC::electron_pairs_1_dy].psiTau;
         
  vector_double& psi2_tau_dx= wavefunctions[WC::electron_pairs_2_dx].psiTau;
  vector_double& psi2_tau_dy= wavefunctions[WC::electron_pairs_2_dy].psiTau;

  for (int ip = 0; ip < electron_pairs; ip++) {
    auto dr = electron_pair_list->pos1[ip] - electron_pair_list->pos2[ip];
    for (int im = iocc1, idx = ip * ivir2 + iocc1; im < iocc2; ++im, ++idx) {
      psi1_tau_dx[idx]  = dr[0] * psi1_dx[idx];
      psi1_tau_dx[idx] += dr[1] * psi1_dy[idx];
      psi1_tau_dx[idx] += dr[2] * psi1_dz[idx];
      psi2_tau_dx[idx]  = dr[0] * psi2_dx[idx];
      psi2_tau_dx[idx] += dr[1] * psi2_dy[idx];
      psi2_tau_dx[idx] += dr[2] * psi2_dz[idx];
    }
  }

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
      ds_p11[io * electrons + jo] = 0.0;
      ds_p12[io * electrons + jo] = 0.0;
      ds_p21[io * electrons + jo] = 0.0;
      ds_p22[io * electrons + jo] = 0.0;
      for (int im = iocc1; im < iocc2; ++im) {
        ds_p11[io * electrons + jo] = ds_p11[io * electrons + jo] + psi[io * lda + im] * (dr[0] * psi_dx[io * lda + im] + dr[1] * psi_dy[io * lda + im] + dr[2] * psi_dz[io * lda + im]);
        ds_p12[io * electrons + jo] = ds_p12[io * electrons + jo] + psi[io * lda + im] * (dr[0] * psi_dx[jo * lda + im] + dr[1] * psi_dy[jo * lda + im] + dr[2] * psi_dz[jo * lda + im]);
        ds_p21[io * electrons + jo] = ds_p21[io * electrons + jo] + psi[jo * lda + im] * (dr[0] * psi_dx[io * lda + im] + dr[1] * psi_dy[io * lda + im] + dr[2] * psi_dz[io * lda + im]);
        ds_p22[io * electrons + jo] = ds_p22[io * electrons + jo] + psi[jo * lda + im] * (dr[0] * psi_dx[jo * lda + im] + dr[1] * psi_dy[jo * lda + im] + dr[2] * psi_dz[jo * lda + im]);
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
    ds_p11[io * electrons + io] = 0.0;
    ds_p12[io * electrons + io] = 0.0;
    ds_p21[io * electrons + io] = 0.0;
    ds_p22[io * electrons + io] = 0.0;
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
  auto iocc1 = electron_psi.iocc1;
  auto iocc2 = electron_psi.iocc2;
  auto ivir1 = electron_psi.ivir1;
  auto ivir2 = electron_psi.ivir2;
  blas_wrapper.dsyrk(BLAS_WRAPPER::FILL_FULL, true,
      electrons, iocc2 - iocc1,
      1.0,
      electron_psi.psi, iocc1, electron_psi.lda,
      0.0,
      op12, 0, electrons);

  blas_wrapper.dcopy(electrons, op12, electrons+1, op11, 1);

  blas_wrapper.dsyrk(BLAS_WRAPPER::FILL_FULL, true,
      electrons, ivir2 - ivir1,
      1.0,
      electron_psi.psi, ivir1, electron_psi.lda,
      0.0,
      ov12, 0, electrons);

  blas_wrapper.dcopy(op12.size(), op12, 1, ok12, 1);
  blas_wrapper.dsyrk(BLAS_WRAPPER::FILL_FULL, true,
      electrons, iocc1,
      1.0,
      electron_psi.psi, electron_psi.lda,
      1.0,
      ok12, electrons);

  blas_wrapper.dscal(electrons, 0.0, op12, electrons+1);
  blas_wrapper.dscal(electrons, 0.0, ov12, electrons+1);
  blas_wrapper.dscal(electrons, 0.0, ok12, electrons+1);
}

void F12_Traces::build_two_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2) {
  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      electron_pair_psi1.psi, iocc1, ivir2,
      electron_pair_psi1.psi, iocc1, ivir2,
      p11, 1);
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      electron_pair_psi1.psi, iocc1, ivir2,
      electron_pair_psi2.psi, iocc1, ivir2,
      p12, 1);
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      electron_pair_psi2.psi, iocc1, ivir2,
      electron_pair_psi2.psi, iocc1, ivir2,
      p22, 1);
  blas_wrapper.batched_ddot(electron_pairs, iocc1,
      electron_pair_psi1.psi, 0, ivir2,
      electron_pair_psi2.psi, 0, ivir2,
      k12, 1);
  blas_wrapper.plus(k12.begin(), k12.end(), p12.begin(), k12.begin());
}

void F12_Traces::build_two_e_one_e_traces(const Wavefunction_Type& electron_pair_psi1, const Wavefunction_Type& electron_pair_psi2, const Wavefunction_Type& electron_psi) {
  double alpha = 1.0;
  double beta = 0.0;
  auto iocc1 = electron_psi.iocc1;
  auto iocc2 = electron_psi.iocc2;
  auto ivir1 = electron_psi.ivir1;
  auto ivir2 = electron_psi.ivir2;

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc2 - iocc1,
      alpha,
      electron_psi.psi, iocc1, electron_psi.lda,
      electron_pair_psi1.psi, iocc1, electron_pair_psi2.lda,
      beta,
      p13, 0, electrons);

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc2 - iocc1,
      alpha,
      electron_psi.psi, iocc1, electron_psi.lda,
      electron_pair_psi2.psi, iocc1, electron_pair_psi2.lda,
      beta,
      p23, 0, electrons);

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, ivir2 - ivir1,
      alpha,
      electron_psi.psi, ivir1, electron_psi.lda,
      electron_pair_psi1.psi, ivir1, electron_pair_psi2.lda,
      beta,
      v13, 0, electrons);

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, ivir2 - ivir1,
      alpha,
      electron_psi.psi, ivir1, electron_psi.lda,
      electron_pair_psi2.psi, ivir1, electron_pair_psi2.lda,
      beta,
      v23, 0, electrons);

  beta = 1.0;
  blas_wrapper.dcopy(p13.size(), p13, 1, k13, 1);
  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc1,
      alpha,
      electron_psi.psi, electron_psi.lda,
      electron_pair_psi1.psi, electron_pair_psi2.lda,
      beta,
      k13, electrons);

  blas_wrapper.dcopy(p23.size(), p23, 1, k23, 1);
  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc1,
      alpha,
      electron_psi.psi, electron_psi.lda,
      electron_pair_psi2.psi, electron_pair_psi2.lda,
      beta,
      k23, electrons);
}

void F12_Traces::build_two_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list) {
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;

  const vector_double& psi1 = wavefunctions[WC::electron_pairs_1].psi;
  const vector_double& psi2 = wavefunctions[WC::electron_pairs_2].psi;
  const vector_double& psi1_tau_dx = wavefunctions[WC::electron_pairs_1_dx].psiTau;
  const vector_double& psi2_tau_dx = wavefunctions[WC::electron_pairs_2_dx].psiTau;

  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      psi1, iocc1, ivir2,
      psi1_tau_dx, iocc1, ivir2,
      dp11, 1);
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      psi1, iocc1, ivir2,
      psi2_tau_dx, iocc1, ivir2,
      dp12, 1); 
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      psi2, iocc1, ivir2,
      psi1_tau_dx, iocc1, ivir2,
      dp21, 1);
  blas_wrapper.batched_ddot(electron_pairs, iocc2 - iocc1,
      psi2, iocc1, ivir2,
      psi2_tau_dx, iocc1, ivir2,
      dp22, 1);
}

void F12_Traces::build_two_e_one_e_derivative_traces(std::unordered_map<int, Wavefunction_Type>& wavefunctions, const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  auto iocc1 = wavefunctions[WC::electrons].iocc1;
  auto iocc2 = wavefunctions[WC::electrons].iocc2;
  auto ivir1 = wavefunctions[WC::electrons].ivir1;
  auto ivir2 = wavefunctions[WC::electrons].ivir2;
  auto o_lda = wavefunctions[WC::electrons].lda;
  auto p_lda = wavefunctions[WC::electron_pairs_1].lda;

  const vector_double& psi = wavefunctions[WC::electrons].psi;
  const vector_double& psi1_tau_dx = wavefunctions[WC::electron_pairs_1_dx].psiTau;
  const vector_double& psi2_tau_dx = wavefunctions[WC::electron_pairs_2_dx].psiTau;

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc2 - iocc1,
      1.0,
      psi, iocc1, o_lda,
      psi1_tau_dx, iocc1, p_lda,
      0.0,
      dp31, 0, electrons);

  blas_wrapper.dgemm(true, false,
      electrons, electron_pairs, iocc2 - iocc1,
      1.0,
      psi, iocc1, o_lda,
      psi2_tau_dx, iocc1, p_lda,
      0.0,
      dp32, 0, electrons);
}

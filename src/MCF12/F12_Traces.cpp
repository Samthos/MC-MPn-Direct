//
// Created by aedoran on 1/2/20.
//

#include <algorithm>
#include <numeric>
#include <array>

#include "F12_Traces.h"

F12_Traces::F12_Traces(int io1, int io2, int iv1, int iv2, int electron_pairs_, int electrons_) :
    iocc1(io1),
    iocc2(io2),
    ivir1(iv1),
    ivir2(iv2),
    electron_pairs(electron_pairs_),
    electrons(electrons_),
    op11(electrons, 0.0),
    op12(electrons, std::vector<double>(electrons, 0.0)),
    ok12(electrons, std::vector<double>(electrons, 0.0)),
    ov12(electrons, std::vector<double>(electrons, 0.0)),
    dop11(electrons, std::vector<double>(electrons, 0.0)),
    dop12(electrons, std::vector<double>(electrons, 0.0)),
    p11(electron_pairs, 0.0),
    p12(electron_pairs, 0.0),
    p22(electron_pairs, 0.0),
    k12(electron_pairs, 0.0),
    dp11(electron_pairs, 0.0),
    dp12(electron_pairs, 0.0),
    dp21(electron_pairs, 0.0),
    dp22(electron_pairs, 0.0),
    p13(electron_pairs, std::vector<double>(electrons, 0.0)),
    k13(electron_pairs, std::vector<double>(electrons, 0.0)),
    v13(electron_pairs, std::vector<double>(electrons, 0.0)),
    dp31(electron_pairs, std::vector<double>(electrons, 0.0)),
    p23(electron_pairs, std::vector<double>(electrons, 0.0)),
    k23(electron_pairs, std::vector<double>(electrons, 0.0)),
    v23(electron_pairs, std::vector<double>(electrons, 0.0)),
    dp32(electron_pairs, std::vector<double>(electrons, 0.0)),
    ds_p11(electrons, std::vector<double>(electrons, 0.0)),
    ds_p12(electrons, std::vector<double>(electrons, 0.0)),
    ds_p21(electrons, std::vector<double>(electrons, 0.0)),
    ds_p22(electrons, std::vector<double>(electrons, 0.0)),
    ds_p31(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0))),
    ds_p32(electrons, std::vector<std::vector<double>>(electrons, std::vector<double>(electrons, 0.0)))
{
}

void F12_Traces::update_v(std::unordered_map<int, Wavefunction>& wavefunctions) {
  build_one_e_traces(wavefunctions[electrons]);
  build_one_e_one_e_traces(wavefunctions[electrons]);
  build_two_e_traces(wavefunctions[electron_pairs_1], wavefunctions[electron_pairs_2]);
  build_two_e_one_e_traces(wavefunctions[electron_pairs_1], wavefunctions[electron_pairs_2], wavefunctions[electrons]);
}

void F12_Traces::update_bx(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  build_two_e_derivative_traces(wavefunctions, electron_pair_list);
  build_two_e_one_e_derivative_traces(wavefunctions, electron_pair_list, electron_list);
}

void F12_Traces::update_bx_fd_traces(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_List* electron_list) {
  std::array<double, 3> dr{};

  auto lda = wavefunctions[electrons].lda;
  const double* psi = wavefunctions[electrons].data();
  const double* psi_dx = wavefunctions[electrons_dx].data();
  const double* psi_dy = wavefunctions[electrons_dy].data();
  const double* psi_dz = wavefunctions[electrons_dz].data();

  for (int io = 0; io < electrons; ++io) {
    for (int jo = 0; jo < electrons; ++jo) {
      std::transform(electron_list->pos[io].begin(), electron_list->pos[io].end(), electron_list->pos[jo].begin(), dr.begin(), std::minus<>());
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
  }
}

void F12_Traces::build_one_e_traces(const Wavefunction& electron_psi) {
  for(int io = 0; io < electrons;io++) {
    op11[io] = std::inner_product(electron_psi.data() + io * electron_psi.lda + iocc1, 
        electron_psi.data() + io * electron_psi.lda + iocc2, 
        electron_psi.data() + io * electron_psi.lda + iocc1, 
        0.0);
  }
}

void F12_Traces::build_one_e_one_e_traces(const Wavefunction& electron_psi) {
  for(int io = 0; io < electrons;io++) {
    for(int jo = 0; jo < electrons;jo++) {
      if (jo != io) {
        op12[io][jo] = std::inner_product(electron_psi.data() + io * electron_psi.lda + iocc1,
            electron_psi.data() + io * electron_psi.lda + iocc2,
            electron_psi.data() + jo * electron_psi.lda + iocc1,
            0.0);

        ok12[io][jo] = std::inner_product(electron_psi.data() + io * electron_psi.lda        ,
            electron_psi.data() + io * electron_psi.lda + iocc1,
            electron_psi.data() + jo * electron_psi.lda,
            op12[io][jo]);

        ov12[io][jo] = std::inner_product(electron_psi.data() + io * electron_psi.lda + ivir1,
            electron_psi.data() + io * electron_psi.lda + ivir2,
            electron_psi.data() + jo * electron_psi.lda + ivir1,
            0.0);
      }
    }
  }
}

void F12_Traces::build_two_e_traces(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2) {
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

void F12_Traces::build_two_e_one_e_traces(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi) {
  for(int ip = 0; ip < electron_pairs; ++ip) {
    for(int io = 0; io < electrons; ++io) {
      p13[ip][io] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
          electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc2,
          electron_psi.data() + io * electron_psi.lda + iocc1,
          0.0);
      p23[ip][io] = std::inner_product(electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc1,
          electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc2,
          electron_psi.data() + io * electron_psi.lda + iocc1,
          0.0);

      k13[ip][io] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda,
          electron_pair_psi1.data() + ip * electron_pair_psi1.lda + iocc1,
          electron_psi.data() + io * electron_psi.lda,
          p13[ip][io]);
      k23[ip][io] = std::inner_product(electron_pair_psi2.data() + ip * electron_pair_psi2.lda,
          electron_pair_psi2.data() + ip * electron_pair_psi2.lda + iocc1,
          electron_psi.data() + io * electron_psi.lda,
          p23[ip][io]);

      v13[ip][io] = std::inner_product(electron_pair_psi1.data() + ip * electron_pair_psi1.lda  + ivir1,
          electron_pair_psi1.data() + ip * electron_pair_psi1.lda + ivir2,
          electron_psi.data() + io * electron_psi.lda + ivir1,
          0.0);
      v23[ip][io] = std::inner_product(electron_pair_psi2.data() + ip * electron_pair_psi2.lda  + ivir1,
          electron_pair_psi2.data() + ip * electron_pair_psi2.lda + ivir2,
          electron_psi.data() + io * electron_psi.lda + ivir1,
          0.0);
    }
  }
}

void F12_Traces::build_two_e_derivative_traces(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list) {
  std::array<double, 3> dr{};
  auto lda = wavefunctions[electron_pairs_1].lda;

  const double* psi1 = wavefunctions[electron_pairs_1].data();
  const double* psi2 = wavefunctions[electron_pairs_2].data();

  const double* psi1_dx = wavefunctions[electron_pairs_1_dx].data();
  const double* psi1_dy = wavefunctions[electron_pairs_1_dy].data();
  const double* psi1_dz = wavefunctions[electron_pairs_1_dz].data();

  const double* psi2_dx = wavefunctions[electron_pairs_2_dx].data();
  const double* psi2_dy = wavefunctions[electron_pairs_2_dy].data();
  const double* psi2_dz = wavefunctions[electron_pairs_2_dz].data();

  for (int ip = 0; ip < electron_pairs; ip++) {
    std::transform(electron_pair_list->pos1[ip].begin(), electron_pair_list->pos1[ip].end(), electron_pair_list->pos2[ip].begin(), dr.begin(), std::minus<>());
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

void F12_Traces::build_two_e_one_e_derivative_traces(std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  std::array<double, 3> dr{};

  auto o_lda = wavefunctions[electrons].lda;
  auto p_lda = wavefunctions[electron_pairs_1].lda;

  const double* psi = wavefunctions[electrons].data();

  const double* psi1_dx = wavefunctions[electron_pairs_1_dx].data();
  const double* psi1_dy = wavefunctions[electron_pairs_1_dy].data();
  const double* psi1_dz = wavefunctions[electron_pairs_1_dz].data();

  const double* psi2_dx = wavefunctions[electron_pairs_2_dx].data();
  const double* psi2_dy = wavefunctions[electron_pairs_2_dy].data();
  const double* psi2_dz = wavefunctions[electron_pairs_2_dz].data();

  for(int ip = 0; ip < electron_pairs; ++ip) {
    std::transform(electron_pair_list->pos1[ip].begin(), electron_pair_list->pos1[ip].begin() + 3, electron_pair_list->pos2[ip].begin(), dr.begin(), std::minus<>());
    for(int io = 0; io < electrons; ++io) {
      dp31[ip][io] = 0.0;
      dp32[ip][io] = 0.0;
      for(int im = iocc1, p_idx = ip * p_lda + iocc1, o_idx = io * o_lda + iocc1; im < iocc2; ++im, p_idx++, o_idx++) {
        dp31[ip][io] = dp31[ip][io] + psi[o_idx] * (dr[0] * psi1_dx[p_idx] + dr[1] * psi1_dy[p_idx] + dr[2] * psi1_dz[p_idx]);
        dp32[ip][io] = dp32[ip][io] + psi[o_idx] * (dr[0] * psi2_dx[p_idx] + dr[1] * psi2_dy[p_idx] + dr[2] * psi2_dz[p_idx]);
      }
    }
  }
}

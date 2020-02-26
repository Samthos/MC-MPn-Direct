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

void F12_Traces::update_v(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi) {
  build_one_e_traces(electron_psi);
  build_one_e_one_e_traces(electron_psi);
  build_two_e_traces(electron_pair_psi1, electron_pair_psi2);
  build_two_e_one_e_traces(electron_pair_psi1, electron_pair_psi2, electron_psi);
}

/*
void F12_Traces::update_bx(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi) {
  build_two_e_derivative_traces(electron_pair_list);
  build_two_e_one_e_derivative_traces(electron_pair_list, el_one_list);
}
*/

/*
void F12_Traces::update_bx_fd_traces(const std::vector<el_one_typ> &const Wavefunction& electron_psi) {
  std::array<double, 3> dr{};
  for (int io = 0; io < electrons; ++io) {
    for (int jo = 0; jo < electrons; ++jo) {
      std::transform(el_one_list[io].pos, el_one_list[io].pos + 3, el_one_list[jo].pos, dr.begin(), std::minus<>());
      ds_p11[io][jo] = 0.0;
      ds_p12[io][jo] = 0.0;
      ds_p21[io][jo] = 0.0;
      ds_p22[io][jo] = 0.0;
      for (int im = iocc1; im < iocc2; ++im) {
        ds_p11[io][jo] = ds_p11[io][jo] + el_one_list[io].psi1[im] * (dr[0] * el_one_list[io].psi1_dx[im] + dr[1] * el_one_list[io].psi1_dy[im] + dr[2] * el_one_list[io].psi1_dz[im]);
        ds_p12[io][jo] = ds_p12[io][jo] + el_one_list[io].psi1[im] * (dr[0] * el_one_list[jo].psi1_dx[im] + dr[1] * el_one_list[jo].psi1_dy[im] + dr[2] * el_one_list[jo].psi1_dz[im]);
        ds_p21[io][jo] = ds_p21[io][jo] + el_one_list[jo].psi1[im] * (dr[0] * el_one_list[io].psi1_dx[im] + dr[1] * el_one_list[io].psi1_dy[im] + dr[2] * el_one_list[io].psi1_dz[im]);
        ds_p22[io][jo] = ds_p22[io][jo] + el_one_list[jo].psi1[im] * (dr[0] * el_one_list[jo].psi1_dx[im] + dr[1] * el_one_list[jo].psi1_dy[im] + dr[2] * el_one_list[jo].psi1_dz[im]);
      }

      for (int ko = 0; ko < electrons; ++ko) {
        ds_p31[io][jo][ko] = 0.0;
        ds_p32[io][jo][ko] = 0.0;
        for (int im = iocc1; im < iocc2; ++im) {
          ds_p31[io][jo][ko] = ds_p31[io][jo][ko] + el_one_list[ko].psi1[im] * (dr[0] * el_one_list[io].psi1_dx[im] + dr[1] * el_one_list[io].psi1_dy[im] + dr[2] * el_one_list[io].psi1_dz[im]);
          ds_p32[io][jo][ko] = ds_p32[io][jo][ko] + el_one_list[ko].psi1[im] * (dr[0] * el_one_list[jo].psi1_dx[im] + dr[1] * el_one_list[jo].psi1_dy[im] + dr[2] * el_one_list[jo].psi1_dz[im]);
        }
      }
    }
  }
}
*/

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

/*
void F12_Traces::build_two_e_derivative_traces(const std::vector<electron_pair_typ> &const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2) {
  std::array<double, 3> dr{};
  for(int ip = 0; ip < electron_pairs;ip++) {
    std::transform(electron_pair_list[ip].pos1, electron_pair_list[ip].pos1 + 3, electron_pair_list[ip].pos2, dr.begin(), std::minus<>());
    dp11[ip] = 0.0;
    dp12[ip] = 0.0;
    dp21[ip] = 0.0;
    dp22[ip] = 0.0;
    for (int im = iocc1; im < iocc2; ++im) {
      dp11[ip] = dp11[ip] + electron_pair_list[ip].psi1[im] * (dr[0] * electron_pair_list[ip].psi1_dx[im] + dr[1] * electron_pair_list[ip].psi1_dy[im] + dr[2] * electron_pair_list[ip].psi1_dz[im]);
      dp12[ip] = dp12[ip] + electron_pair_list[ip].psi1[im] * (dr[0] * electron_pair_list[ip].psi2_dx[im] + dr[1] * electron_pair_list[ip].psi2_dy[im] + dr[2] * electron_pair_list[ip].psi2_dz[im]);
      dp21[ip] = dp21[ip] + electron_pair_list[ip].psi2[im] * (dr[0] * electron_pair_list[ip].psi1_dx[im] + dr[1] * electron_pair_list[ip].psi1_dy[im] + dr[2] * electron_pair_list[ip].psi1_dz[im]);
      dp22[ip] = dp22[ip] + electron_pair_list[ip].psi2[im] * (dr[0] * electron_pair_list[ip].psi2_dx[im] + dr[1] * electron_pair_list[ip].psi2_dy[im] + dr[2] * electron_pair_list[ip].psi2_dz[im]);
    }
  }
}
*/

/*
void F12_Traces::build_two_e_one_e_derivative_traces(const Wavefunction& electron_pair_psi1, const Wavefunction& electron_pair_psi2, const Wavefunction& electron_psi) {
  std::array<double, 3> dr{};
  for(int ip = 0; ip < electron_pairs; ++ip) {
    std::transform(electron_pair_list[ip].pos1, electron_pair_list[ip].pos1 + 3, electron_pair_list[ip].pos2, dr.begin(), std::minus<>());
    for(int io = 0; io < electrons; ++io) {
      dp31[ip][io] = 0.0;
      dp32[ip][io] = 0.0;
      for(int im=iocc1;im<iocc2;++im) {
        dp31[ip][io] = dp31[ip][io] + el_one_list[io].psi1[im]*(dr[0]*electron_pair_list[ip].psi1_dx[im]+dr[1]*electron_pair_list[ip].psi1_dy[im]+dr[2]*electron_pair_list[ip].psi1_dz[im]);
        dp32[ip][io] = dp32[ip][io] + el_one_list[io].psi1[im]*(dr[0]*electron_pair_list[ip].psi2_dx[im]+dr[1]*electron_pair_list[ip].psi2_dy[im]+dr[2]*electron_pair_list[ip].psi2_dz[im]);
      }
    }
  }
}
*/


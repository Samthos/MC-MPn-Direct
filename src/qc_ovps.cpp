#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "blas_calls.h"
#include "qc_ovps.h"

void print_out(double* A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%12.8f", A[i * n + j]);
    }
    printf("\n");
  }
}

void OVPs::init_02(int p1, int p2, int p3, int p4, int p5, const Basis &basis) {
  mc_pair_num = p1;
  numBand = p2;
  offBand = p3;
  numDiff = p4;
  numBlock = p5;

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;
  lambda = 2.0 * (basis.nw_en[ivir1] - basis.nw_en[iocc2 - 1]);

#ifdef QUAD_TAU
  ovps.t_save_val1 = new double[21 * ivir2];
  ovps.tg_save_val1 = new double[21 * numBand];
  ovps.tgc_save_val1 = new double[21 * numBand];
#else
  ovps.t_val1 = new double[ivir2];
  ovps.tg_val1 = new double[ivir2];
  ovps.tgc_val1 = new double[ivir2];
#endif  // QUAD_TAU
}
void OVPs::alloc_02() {
  d_ovps.occ1 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.occ2 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.vir1 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.vir2 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.psi1 = new double[mc_pair_num * (ivir2 - iocc1)];
  d_ovps.psi2 = new double[mc_pair_num * (ivir2 - iocc1)];
  d_ovps.occTau1 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.occTau2 = new double[mc_pair_num * (iocc2 - iocc1)];
  d_ovps.virTau1 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.virTau2 = new double[mc_pair_num * (ivir2 - ivir1)];
  d_ovps.rv = new double[mc_pair_num];

  d_ovps.os_13 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_14 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_23 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_24 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_13 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_14 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_23 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_24 = new double[mc_pair_num * mc_pair_num];

  d_ovps.ps_24 = new double[mc_pair_num * mc_pair_num * numBand];

  d_ovps.en2mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en2pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en2m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en2p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];

  d_ovps.ent = new double[(ivir2 - iocc1) * mc_pair_num];

  d_ovps.en2 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));

  d_ovps.en2Ex1 = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  d_ovps.en2Ex2 = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  d_ovps.en2Block = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  for (uint32_t i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      d_ovps.en2[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      for (uint32_t k = 0; k < d_ovps.en2Ex1[i][j].size(); k++) {
        d_ovps.en2Ex1[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
        d_ovps.en2Ex2[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
        d_ovps.en2Block[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      }
    }
  }
}
void OVPs::free_tau_02() {
#ifdef QUAD_TAU
  delete[] ovps.t_save_val1;
  delete[] ovps.tg_save_val1;
  delete[] ovps.tgc_save_val1;
#else
  delete[] ovps.t_val1;
  delete[] ovps.tg_val1;
  delete[] ovps.tgc_val1;
#endif  // QUAD_TAU
}
void OVPs::free_02() {
  delete[] d_ovps.occ1;
  delete[] d_ovps.occ2;
  delete[] d_ovps.vir1;
  delete[] d_ovps.vir2;
  delete[] d_ovps.psi1;
  delete[] d_ovps.psi2;
  delete[] d_ovps.occTau1;
  delete[] d_ovps.occTau2;
  delete[] d_ovps.virTau1;
  delete[] d_ovps.virTau2;
  delete[] d_ovps.rv;

  delete[] d_ovps.os_13;
  delete[] d_ovps.os_14;
  delete[] d_ovps.os_23;
  delete[] d_ovps.os_24;

  delete[] d_ovps.vs_13;
  delete[] d_ovps.vs_14;
  delete[] d_ovps.vs_23;
  delete[] d_ovps.vs_24;

  delete[] d_ovps.ps_24;

  delete[] d_ovps.ent;
  delete[] d_ovps.en2mCore;
  delete[] d_ovps.en2pCore;
  delete[] d_ovps.en2m;
  delete[] d_ovps.en2p;

  for (uint32_t i = 0; i < d_ovps.en2Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en2Ex1[i].size(); j++) {
      delete[] d_ovps.en2[i][j];
      for (uint32_t k = 0; k < d_ovps.en2Ex1[i][j].size(); k++) {
        delete[] d_ovps.en2Ex1[i][j][k];
        delete[] d_ovps.en2Ex2[i][j][k];
        delete[] d_ovps.en2Block[i][j][k];
      }
    }
  }
}

void OVPs::init_03(int p1, int p2, int p3, int p4, int p5, const Basis &basis) {
  init_02(p1, p2, p3, p4, p5, basis);

#ifdef QUAD_TAU
  ovps.t_save_val2 = new double[21 * ivir2];
  ovps.tg_save_val2 = new double[21 * numBand];
  ovps.tgc_save_val2 = new double[21 * numBand];
#else
  ovps.t_val2 = new double[ivir2];
  ovps.tg_val2 = new double[ivir2];
  ovps.tgc_val2 = new double[ivir2];
#endif  // QUAD_TAU

  ovps.t_val12 = new double[ivir2];
  ovps.tg_val12 = new double[ivir2];
  ovps.tgc_val12 = new double[ivir2];
}
void OVPs::alloc_03() {
  alloc_02();
  d_ovps.os_15 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_16 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_25 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_26 = new double[mc_pair_num * mc_pair_num];

  d_ovps.os_35 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_36 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_45 = new double[mc_pair_num * mc_pair_num];
  d_ovps.os_46 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_15 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_16 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_25 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_26 = new double[mc_pair_num * mc_pair_num];

  d_ovps.vs_35 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_36 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_45 = new double[mc_pair_num * mc_pair_num];
  d_ovps.vs_46 = new double[mc_pair_num * mc_pair_num];

  d_ovps.ps_12c = new double[mc_pair_num * numBand];
  d_ovps.ps_22c = new double[mc_pair_num * numBand];

  d_ovps.en3_1pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12pCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_1mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_2mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12mCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_12cCore = new double[mc_pair_num * mc_pair_num];
  d_ovps.en3_22cCore = new double[mc_pair_num * mc_pair_num];

  d_ovps.en3_1p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_2p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_12p = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_1m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_2m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_12m = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
  d_ovps.en3_c = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];

  d_ovps.en3c12 = new double[mc_pair_num];
  d_ovps.en3c22 = new double[mc_pair_num];
  d_ovps.one = new double[mc_pair_num];
  std::fill(d_ovps.one, d_ovps.one + mc_pair_num, 1.0);

  d_ovps.en3 = std::vector<std::vector<double*>>(numBand, std::vector<double*>(numDiff));
  d_ovps.en3Ex1 = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  d_ovps.en3Ex2 = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  d_ovps.en3Block = std::vector<std::vector<std::vector<double*>>>(numBand, std::vector<std::vector<double*>>(numDiff, std::vector<double*>(numBlock)));
  for (uint32_t i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      d_ovps.en3[i][j] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      for (uint32_t k = 0; k < d_ovps.en3Ex1[i][j].size(); k++) {
        d_ovps.en3Ex1[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
        d_ovps.en3Ex2[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
        d_ovps.en3Block[i][j][k] = new double[(ivir2 - iocc1) * (ivir2 - iocc1)];
      }
    }
  }
}
void OVPs::free_tau_03() {
  free_tau_02();

#ifdef QUAD_TAU
  delete[] ovps.t_save_val2;
  delete[] ovps.tg_save_val2;
  delete[] ovps.tgc_save_val2;
#else
  delete[] ovps.t_val2;
  delete[] ovps.tg_val2;
  delete[] ovps.tgc_val2;
#endif  // QUAD_TAU
  delete[] ovps.t_val12;
  delete[] ovps.tg_val12;
  delete[] ovps.tgc_val12;
}
void OVPs::free_03() {
  free_02();

  delete[] d_ovps.os_15;
  delete[] d_ovps.os_16;
  delete[] d_ovps.os_25;
  delete[] d_ovps.os_26;

  delete[] d_ovps.os_35;
  delete[] d_ovps.os_36;
  delete[] d_ovps.os_45;
  delete[] d_ovps.os_46;

  delete[] d_ovps.vs_15;
  delete[] d_ovps.vs_16;
  delete[] d_ovps.vs_25;
  delete[] d_ovps.vs_26;

  delete[] d_ovps.vs_35;
  delete[] d_ovps.vs_36;
  delete[] d_ovps.vs_45;
  delete[] d_ovps.vs_46;

  delete[] d_ovps.ps_12c;
  delete[] d_ovps.ps_22c;

  delete[] d_ovps.en3_1pCore;
  delete[] d_ovps.en3_2pCore;
  delete[] d_ovps.en3_12pCore;
  delete[] d_ovps.en3_1mCore;
  delete[] d_ovps.en3_2mCore;
  delete[] d_ovps.en3_12mCore;
  delete[] d_ovps.en3_12cCore;
  delete[] d_ovps.en3_22cCore;

  delete[] d_ovps.en3_1p;
  delete[] d_ovps.en3_2p;
  delete[] d_ovps.en3_12p;
  delete[] d_ovps.en3_1m;
  delete[] d_ovps.en3_2m;
  delete[] d_ovps.en3_12m;
  delete[] d_ovps.en3_c;

  delete[] d_ovps.one;
  delete[] d_ovps.en3c12;
  delete[] d_ovps.en3c22;

  for (uint32_t i = 0; i < d_ovps.en3Ex1.size(); i++) {
    for (uint32_t j = 0; j < d_ovps.en3Ex1[i].size(); j++) {
      delete[] d_ovps.en3[i][j];
      for (uint32_t k = 0; k < d_ovps.en3Ex1[i][j].size(); k++) {
        delete[] d_ovps.en3Ex1[i][j][k];
        delete[] d_ovps.en3Ex2[i][j][k];
        delete[] d_ovps.en3Block[i][j][k];
      }
    }
  }
}

void OVPs::zero_energy_arrays_02() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
}
void OVPs::zero_energy_arrays_03() {
  for (auto& it : d_ovps.en2) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
  for (auto& it : d_ovps.en3) {
    for (auto& jt : it) {
      std::fill(jt, jt + ((ivir2 - iocc1) * (ivir2 - iocc1)), 0.0);
      ;
    }
  }
}

void OVPs::new_tau_02(Basis& basis, Random& random) {
  int im, am;
  double en_i, en_a;

  double p = random.get_rand();
  xx1 = -log(1.0 - p) / lambda;
  t1_twgt = 1.0 / (lambda * (1.0 - p));

  p = random.get_rand();
  xx2 = -log(1.0 - p) / lambda;
  t2_twgt = t1_twgt / (lambda * (1.0 - p));

  for (im = iocc1; im < iocc2; im++) {
    en_i = basis.nw_en[im];
    ovps.t_val1[im] = exp(en_i * xx1);
  }
  for (am = ivir1; am < ivir2; am++) {
    en_a = basis.nw_en[am];
    ovps.t_val1[am] = exp(-en_a * xx1);
  }
  for (am = 0; am < numBand; ++am) {
    en_a = basis.nw_en[iocc2 - offBand + am];
    ovps.tg_val1[am] = exp(en_a * xx1);
    ovps.tgc_val1[am] = exp(-en_a * xx1);
  }
}
void OVPs::new_tau_03(Basis& basis, Random& random) {
  int im, am;
  double en_i, en_a;

  double p = random.get_rand();
  xx1 = -log(1.0 - p) / lambda;
  t1_twgt = 1.0 / (lambda * (1.0 - p));

  p = random.get_rand();
  xx2 = -log(1.0 - p) / lambda;
  t2_twgt = t1_twgt / (lambda * (1.0 - p));

  for (im = iocc1; im < iocc2; im++) {
    en_i = basis.nw_en[im];
    ovps.t_val1[im] = exp(en_i * xx1);

    ovps.t_val2[im] = exp(en_i * xx2);
    ovps.t_val12[im] = ovps.t_val1[im] * ovps.t_val2[im];
  }
  for (am = ivir1; am < ivir2; am++) {
    en_a = basis.nw_en[am];
    ovps.t_val1[am] = exp(-en_a * xx1);

    ovps.t_val2[am] = exp(-en_a * xx2);
    ovps.t_val12[am] = ovps.t_val1[am] * ovps.t_val2[am];
  }
  for (am = 0; am < numBand; ++am) {
    en_a = basis.nw_en[iocc2 - offBand + am];
    ovps.tg_val1[am] = exp(en_a * xx1);
    ovps.tgc_val1[am] = exp(-en_a * xx1);

    ovps.tg_val2[am] = exp(en_a * xx2);
    ovps.tgc_val2[am] = exp(-en_a * xx2);
    ovps.tg_val12[am] = ovps.tg_val1[am] * ovps.tg_val2[am];
    ovps.tgc_val12[am] = ovps.tgc_val1[am] * ovps.tgc_val2[am];
  }
}

void OVPs::set_tau_02(int t1) {
  static std::array<double, 21> tauWgt = {
      1240.137264162088286,
      0.005872796340197,
      95.637046659066982,
      0.016712318843111,
      22.450252490071218,
      0.029395130776845,
      8.242542564370559,
      0.043145326054643,
      3.876926315587520,
      0.058729927034166,
      2.128605721895366,
      0.077568088880637,
      1.291883267060868,
      0.101131338617294,
      0.839203153977961,
      0.131127958307208,
      0.573533895185722,
      0.170432651403897,
      0.407885334132820,
      0.223862010922047,
      0.298891108005834};
  static std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};
  ovps.t_val1 = ovps.t_save_val1 + t1 * ivir2;
  ovps.tg_val1 = ovps.tg_save_val1 + t1 * numBand;
  ovps.tgc_val1 = ovps.tgc_save_val1 + t1 * numBand;

  t1_twgt = tauWgt[t1];
  xx1 = xx[t1];
}
void OVPs::set_tau_03(int t1, int t2) {
  static std::array<double, 21> tauWgt = {
      1240.137264162088286,
      0.005872796340197,
      95.637046659066982,
      0.016712318843111,
      22.450252490071218,
      0.029395130776845,
      8.242542564370559,
      0.043145326054643,
      3.876926315587520,
      0.058729927034166,
      2.128605721895366,
      0.077568088880637,
      1.291883267060868,
      0.101131338617294,
      0.839203153977961,
      0.131127958307208,
      0.573533895185722,
      0.170432651403897,
      0.407885334132820,
      0.223862010922047,
      0.298891108005834};
  static std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};

  ovps.t_val1 = ovps.t_save_val1 + t1 * ivir2;
  ovps.tg_val1 = ovps.tg_save_val1 + t1 * numBand;
  ovps.tgc_val1 = ovps.tgc_save_val1 + t1 * numBand;

  t1_twgt = tauWgt[t1];
  xx1 = xx[t1];

  ovps.t_val2 = ovps.t_save_val2 + t2 * ivir2;
  ovps.tg_val2 = ovps.tg_save_val2 + t2 * numBand;
  ovps.tgc_val2 = ovps.tgc_save_val2 + t2 * numBand;

  std::transform(ovps.t_val1, ovps.t_val1 + ivir2, ovps.t_val2, ovps.t_val12, std::multiplies<double>());
  std::transform(ovps.tg_val1, ovps.tg_val1 + numBand, ovps.tg_val2, ovps.tg_val12, std::multiplies<double>());
  std::transform(ovps.tgc_val1, ovps.tgc_val1 + numBand, ovps.tgc_val2, ovps.tgc_val12, std::multiplies<double>());

  t2_twgt = tauWgt[t1] * tauWgt[t2];
  xx2 = xx[t1];
}

void OVPs::init_tau_02(Basis& basis) {
  std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};

  for (uint it = 0; it < xx.size(); it++) {
    for (int jt = 0; jt < iocc2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(en * xx[it]);
    }
    for (int jt = ivir1; jt < ivir2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(-en * xx[it]);
    }
    for (int jt = 0; jt < numBand; ++jt) {
      double en = basis.nw_en[iocc2 - offBand + jt];
      ovps.tg_save_val1[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val1[it * numBand + jt] = exp(-en * xx[it]);
    }
  }
}
void OVPs::init_tau_03(Basis& basis) {
  std::array<double, 21> xx = {
      459.528454529921248195023509,
      0.002176143805986910199912,
      75.647524700428292021570087,
      0.013219203192174486943822,
      27.635855710538834273393149,
      0.036184875564343521592292,
      13.821771900816584022209099,
      0.072349623997261858221464,
      8.124825510985218102177896,
      0.123079566280893559770959,
      5.238489369094648573366158,
      0.190894727380696543894700,
      3.574116946388957050118051,
      0.279789389938773946919781,
      2.529798344872996818111233,
      0.395288423690625334572246,
      1.834438449215696431693345,
      0.545125948721552511244681,
      1.349829280916060136874535,
      0.740834425610734315092998,
      1.000000000000000000000000};

  for (uint it = 0; it < xx.size(); it++) {
    for (int jt = 0; jt < iocc2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(en * xx[it]);
      ovps.t_save_val2[it * ivir2 + jt] = exp(en * xx[it]);
    }
    for (int jt = ivir1; jt < ivir2; jt++) {
      double en = basis.nw_en[jt];
      ovps.t_save_val1[it * ivir2 + jt] = exp(-en * xx[it]);
      ovps.t_save_val2[it * ivir2 + jt] = exp(-en * xx[it]);
    }
    for (int jt = 0; jt < numBand; ++jt) {
      double en = basis.nw_en[iocc2 - offBand + jt];
      ovps.tg_save_val1[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val1[it * numBand + jt] = exp(-en * xx[it]);
      ovps.tg_save_val2[it * numBand + jt] = exp(en * xx[it]);
      ovps.tgc_save_val2[it * numBand + jt] = exp(-en * xx[it]);
    }
  }
}

void freq_indp_gf(OVPS_ARRAY ovps, int mc_pair_num, int iocc2, int offBand, int numBand) {
  int tidx, tidy, index, lookup;
  for (tidx = 0; tidx < mc_pair_num; tidx++) {
    for (tidy = 0; tidy < numBand; tidy++) {
      index = tidy * mc_pair_num + tidx;
      if (tidy - offBand < 0) {
        lookup = (iocc2 - offBand + tidy) * mc_pair_num + tidx;
        ovps.ps_12c[index] = ovps.occ1[lookup] * ovps.occ2[lookup];
        ovps.ps_22c[index] = ovps.occ2[lookup] * ovps.occ2[lookup];
      } else {
        lookup = (tidy - offBand) * mc_pair_num + tidx;
        ovps.ps_12c[index] = ovps.vir1[lookup] * ovps.vir2[lookup];
        ovps.ps_22c[index] = ovps.vir2[lookup] * ovps.vir2[lookup];
      }
    }
  }
}

void OVPs::update_ovps_02(el_pair_typ* el_pair_list) {
  int ip, am;
  double alpha = 1.00;
  double beta = 0.00;

  //copy weights from el_pair_list to host arrays
  for (ip = 0; ip < mc_pair_num; ip++) {
    if (el_pair_list[ip].is_new) {
      d_ovps.rv[ip] = el_pair_list[ip].rv;
      for (am = iocc1; am < iocc2; am++) {
        d_ovps.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
        d_ovps.occ1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.occ2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
      }
      for (am = ivir1; am < ivir2; am++) {
        d_ovps.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
        d_ovps.vir1[(am - ivir1) * mc_pair_num + ip] = el_pair_list[ip].psi1[am];
        d_ovps.vir2[(am - ivir1) * mc_pair_num + ip] = el_pair_list[ip].psi2[am];
      }
    }
  }
  //copy wave functions from host to device;
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &ovps.t_val1[iocc1], 1, d_ovps.occTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &ovps.t_val1[iocc1], 1, d_ovps.occTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_13, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_24, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_23, mc_pair_num);
  Transpose(d_ovps.os_23, mc_pair_num, d_ovps.os_14);

  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &ovps.t_val1[ivir1], 1, d_ovps.virTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &ovps.t_val1[ivir1], 1, d_ovps.virTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_13, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_24, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_23, mc_pair_num);
  Transpose(d_ovps.vs_23, mc_pair_num, d_ovps.vs_14);

  std::fill(d_ovps.ps_24, d_ovps.ps_24 + mc_pair_num * mc_pair_num * numBand, 0.0);
  for (am = 0; am < numBand; am++) {
    if (am - offBand < 0) {  //construct ps_?? and ps_??c for occupied orbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num, alpha, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.occ2 + (am + iocc2 - iocc1 - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    } else {  //construct ps_?? and ps_??c for virtualorbitals
      cblas_dger(CblasColMajor, mc_pair_num, mc_pair_num, alpha, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.vir2 + (am - offBand) * mc_pair_num, 1, d_ovps.ps_24 + am * mc_pair_num * mc_pair_num, mc_pair_num);
    }
  }
}
void OVPs::update_ovps_03(el_pair_typ* el_pair_list) {
  double alpha = 1.00;
  double beta = 0.00;

  update_ovps_02(el_pair_list);

  freq_indp_gf(d_ovps, mc_pair_num, iocc2 - iocc1, offBand, numBand);

  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &ovps.t_val2[iocc1], 1, d_ovps.occTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &ovps.t_val2[iocc1], 1, d_ovps.occTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_35, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_46, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_45, mc_pair_num);
  Transpose(d_ovps.os_45, mc_pair_num, d_ovps.os_36);


  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &ovps.t_val2[ivir1], 1, d_ovps.virTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &ovps.t_val2[ivir1], 1, d_ovps.virTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_35, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_46, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_45, mc_pair_num);
  Transpose(d_ovps.vs_45, mc_pair_num, d_ovps.vs_36);

  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ1, mc_pair_num, &ovps.t_val12[iocc1], 1, d_ovps.occTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, iocc2 - iocc1, d_ovps.occ2, mc_pair_num, &ovps.t_val12[iocc1], 1, d_ovps.occTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ1, mc_pair_num, beta, d_ovps.os_15, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau2, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_26, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, iocc2 - iocc1, alpha, d_ovps.occTau1, mc_pair_num, d_ovps.occ2, mc_pair_num, beta, d_ovps.os_25, mc_pair_num);
  Transpose(d_ovps.os_25, mc_pair_num, d_ovps.os_16);

  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir1, mc_pair_num, &ovps.t_val12[ivir1], 1, d_ovps.virTau1, mc_pair_num);
  Ddgmm(DDGMM_SIDE_RIGHT, mc_pair_num, ivir2 - ivir1, d_ovps.vir2, mc_pair_num, &ovps.t_val12[ivir1], 1, d_ovps.virTau2, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir1, mc_pair_num, beta, d_ovps.vs_15, mc_pair_num);
  cblas_dgemm_sym(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau2, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_26, mc_pair_num);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mc_pair_num, mc_pair_num, ivir2 - ivir1, alpha, d_ovps.virTau1, mc_pair_num, d_ovps.vir2, mc_pair_num, beta, d_ovps.vs_25, mc_pair_num);
  Transpose(d_ovps.vs_25, mc_pair_num, d_ovps.vs_16);
}

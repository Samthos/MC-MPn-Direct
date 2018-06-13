#include <algorithm>
#include <iostream>
#include <vector>

#include "../qc_monte.h"

void GF::mc_gf_statistics(int step,
                                std::vector<std::vector<double>>& qep,
                                std::vector<std::vector<double*>>& en,
                                std::vector<std::vector<double*>>& enEx1,
                                std::vector<std::vector<double*>>& enEx2) {
  for (auto band = 0; band < iops.iopns[KEYS::NUM_BAND]; band++) {
    auto offset = (ivir2 - iocc1) * (iocc2 - iocc1 - offBand + band) + (iocc2 - iocc1 - offBand + band);
    for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
      qep[band][diff] = en[band][diff][offset];

      // enEx1[i] = en / step + (step-1)*enEx1[i]/(step) i.e update first momenT
      auto alpha = 1.0 / static_cast<double>(step);
      auto beta = (static_cast<double>(step) - 1.0) / static_cast<double>(step);
      std::transform(
          en[band][diff],
          en[band][diff] + ((ivir2 - iocc1) * (ivir2 - iocc1)),
          enEx1[band][diff],
          enEx1[band][diff],
          [&](double a, double b) { return alpha * a + beta * b; });

      // en[i][j] = en[i][j]^2
      std::transform(en[band][diff], en[band][diff] + ((ivir2 - iocc1) * (ivir2 - iocc1)), en[band][diff],
                     [](double a) { return a * a; });

      // enEx2 = en/step + (step-1)*enEx2/(step) i.e update second momenT
      std::transform(
          en[band][diff],
          en[band][diff] + ((ivir2 - iocc1) * (ivir2 - iocc1)),
          enEx2[band][diff],
          enEx2[band][diff],
          [&](double a, double b) { return alpha * a + beta * b; });
    }
  }
}

void GF::mc_gf2_statistics(int band, int step) {
  double diffMultiplier = 1.0;
  double alpha, beta;

  for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
    // en2 = diffmultiplier * en2pCore + en2
    alpha = diffMultiplier;
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en2p,
        ovps.d_ovps.en2p + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en2[band][diff],
        ovps.d_ovps.en2[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en2 = (-1)^i diffMultiplier * en2mCore + en2
    if (diff % 2 == 1) {
      alpha = -diffMultiplier;
    } else {
      alpha = diffMultiplier;
    }
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en2m,
        ovps.d_ovps.en2m + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en2[band][diff],
        ovps.d_ovps.en2[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    diffMultiplier *= tau.get_tau(0);
  }
}

void GF::mc_gf3_statistics(int band, int step) {
  std::array<double, 3> diffMultiplier;
  double alpha;
  double beta;

  diffMultiplier.fill(1.0);

  for (auto diff = 0; diff < iops.iopns[KEYS::DIFFS]; diff++) {
    // en3 = diffMultipllier[0] en3_1mCore
    alpha = diffMultiplier[0];
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_1p,
        ovps.d_ovps.en3_1p + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en3 = diffMultipllier[0] en3_2mCore + en3
    alpha = diffMultiplier[1];
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_2p,
        ovps.d_ovps.en3_2p + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en3 = diffMultipllier[0] en3_12mCore + en3
    alpha = diffMultiplier[2];
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_12p,
        ovps.d_ovps.en3_12p + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en3 = en3_1mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[0];
    } else {
      alpha = diffMultiplier[0];
    }
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_1m,
        ovps.d_ovps.en3_1m + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en3 = en3_2mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[1];
    } else {
      alpha = diffMultiplier[1];
    }
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_2m,
        ovps.d_ovps.en3_2m + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    // en3 = en3_12mCore diffMultiplier + en3
    if (diff % 2 == 1) {
      alpha = -diffMultiplier[2];
    } else {
      alpha = diffMultiplier[2];
    }
    beta = 1.0;
    std::transform(
        ovps.d_ovps.en3_12m,
        ovps.d_ovps.en3_12m + ((ivir2 - iocc1) * (ivir2 - iocc1)),
        ovps.d_ovps.en3[band][diff],
        ovps.d_ovps.en3[band][diff],
        [&](double a, double b) { return alpha * a + beta * b; });

    if (diff == 0) {
      // en3 = en3_c + en3
      alpha = 1.0;
      beta = 1.0;
      std::transform(
          ovps.d_ovps.en3_c,
          ovps.d_ovps.en3_c + ((ivir2 - iocc1) * (ivir2 - iocc1)),
          ovps.d_ovps.en3[band][diff],
          ovps.d_ovps.en3[band][diff],
          [&](double a, double b) { return alpha * a + beta * b; });
    }
    diffMultiplier[0] *= tau.get_tau(0);
    diffMultiplier[1] *= tau.get_tau(1);
    diffMultiplier[2] *= diffMultiplier[0] * diffMultiplier[1];
  }
}

void GF::mc_gf_copy(std::vector<double>& ex1, std::vector<double>& ex2, double* d_ex1, double* d_ex2) {
  std::copy(d_ex1, d_ex1 + ex1.size(), ex1.begin());
  std::copy(d_ex2, d_ex2 + ex2.size(), ex2.begin());
}

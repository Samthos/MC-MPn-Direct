#include <algorithm>
#include <iostream>
#include <vector>
#include "cblas.h"
#include "blas_calls.h"
#include "../qc_monte.h"

void GF::mc_gf_statistics(int step,
    std::vector<std::vector<std::vector<double>>>& enBlock,
    std::vector<std::vector<double>>& enEx1,
    std::vector<std::vector<std::vector<double>>>& enCov) {
  for (auto band = 0; band < iops.iopns[KEYS::NUM_BAND]; band++) {

    uint32_t block = 0;
    uint32_t blockPower2 = 1;

    // enEx1[i] = en / step + (step-1)*enEx1[i]/(step) i.e update first moment
    std::transform(
        enBlock[band][block].begin(),
        enBlock[band][block].end(),
        enEx1[band].begin(),
        enEx1[band].begin(),
        std::plus<>());

    while ((step & (blockPower2-1)) == 0 && block < enBlock[band].size()) {
      if (block < enBlock[band].size()-1) {
        std::transform(
            enBlock[band][block].begin(),
            enBlock[band][block].end(),
            enBlock[band][block+1].begin(),
            enBlock[band][block+1].begin(),
            [](double a, double b) {return 0.5*a + b;});
      }

      // enCov = en/step + (step-1)*enCov/(step) i.e update second moment
      dspr_batched((ivir2-iocc1) * (ivir2-iocc1), iops.iopns[KEYS::DIFFS], 1.0, enBlock[band][block].data(), enCov[band][block].data());

      // zero block;
      std::fill(enBlock[band][block].begin(), enBlock[band][block].end(), 0.0);

      block++;
      blockPower2 *= 2;
    }
  }
}

void GF::mc_gf_full_diffs(int band, std::vector<double> m) {
  double diffMultiplier[iops.iopns[KEYS::DIFFS] * m.size()];
  std::fill(diffMultiplier, diffMultiplier + m.size(), 1.0);
  for (auto col = 1; col < iops.iopns[KEYS::DIFFS]; col++) {
    for (auto row = 0ul; row < m.size(); row++) {
      diffMultiplier[col * m.size() + row] = diffMultiplier[(col-1)*m.size()+row] * m[row];
    }
  }

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      (ivir2-iocc1) * (ivir2-iocc1), iops.iopns[KEYS::DIFFS], m.size(),
      1.0,
      d_ovps.enGrouped.data(), (ivir2-iocc1) * (ivir2-iocc1),
      diffMultiplier, m.size(),
      0.0,
      d_ovps.enBlock[band][0].data(), (ivir2-iocc1) * (ivir2-iocc1));
}

void GF::mc_gf_copy(std::vector<double>& ex, std::vector<double>& d_ex) {
  std::copy(d_ex.begin(), d_ex.end(), ex.begin());
}

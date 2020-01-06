#include <algorithm>
#include <iostream>
#include <vector>
#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"

void GF::mc_gf_statistics(int step,
    std::vector<std::vector<double*>>& enBlock,
    std::vector<double*>& enEx1,
    std::vector<std::vector<double*>>& enCov) {
  /*
  auto mode = std::ios::app;
  if (step == 1) {
    mode = std::ios::trunc;
  }
  std::ofstream os("trajectory_dump", mode);
  os.write((char*) ovps.d_ovps.enBlock[0][0], sizeof(double) * (basis.ivir2 - basis.iocc1) * (basis.ivir2 - basis.iocc1));
   */
  for (auto band = 0; band < iops.iopns[KEYS::NUM_BAND]; band++) {

    uint32_t block = 0;
    uint32_t blockPower2 = 1;

    // enEx1[i] = en / step + (step-1)*enEx1[i]/(step) i.e update first moment
    std::transform(
        enBlock[band][block],
        enBlock[band][block] + iops.iopns[KEYS::DIFFS] * (ivir2-iocc1) * (ivir2-iocc1),
        enEx1[band],
        enEx1[band],
        std::plus<>());

    while ((step & (blockPower2-1)) == 0 && block < enBlock[band].size()) {
      if (block < enBlock[band].size()-1) {
        std::transform(
            enBlock[band][block],
            enBlock[band][block] + iops.iopns[KEYS::DIFFS] * (ivir2-iocc1) * (ivir2-iocc1),
            enBlock[band][block+1],
            enBlock[band][block+1],
            [](double a, double b) {return 0.5*a + b;});
      }

      // enCov = en/step + (step-1)*enCov/(step) i.e update second moment
      dspr_batched((ivir2-iocc1) * (ivir2-iocc1), iops.iopns[KEYS::DIFFS], 1.0, enBlock[band][block], enCov[band][block]);

      // zero block;
      std::fill(enBlock[band][block], enBlock[band][block] +  iops.iopns[KEYS::DIFFS] * (ivir2-iocc1) * (ivir2-iocc1), 0.0);

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
      ovps.d_ovps.enGrouped, (ivir2-iocc1) * (ivir2-iocc1),
      diffMultiplier, m.size(),
      0.0,
      ovps.d_ovps.enBlock[band][0], (ivir2-iocc1) * (ivir2-iocc1));
}

void GF::mc_gf_copy(std::vector<double>& ex, double* d_ex) {
  std::copy(d_ex, d_ex + ex.size(), ex.begin());
}

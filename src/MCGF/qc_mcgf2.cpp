#include "../blas_calls.h"
#include "../qc_monte.h"

void gf2_core(OVPS_ARRAY ovps, int mc_pair_num) {
  int tidx, tidy;
#define TIDX_CONTROL for (tidx = 0; tidx < mc_pair_num; tidx++)
#define TIDY_CONTROL for (tidy = 0; tidy < mc_pair_num; tidy++)
#include "gf2_core.h"
#undef TIDX_CONTROL
#undef TIDY_CONTROL
}

void QC_monte::mcgf2_local_energy_core() {
  gf2_core(ovps.d_ovps, iops.iopns[KEYS::MC_NPAIR]);
}

void QC_monte::mcgf2_local_energy(std::vector<double>& egf2, int band) {
  int nsamp;
  double en2, en2p, en2m;

  en2p = cblas_ddot(iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR],
                    ovps.d_ovps.en2pCore, 1,
                    ovps.d_ovps.ps_24 + band * iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR], 1);
  en2m = cblas_ddot(iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR],
                    ovps.d_ovps.en2mCore, 1,
                    ovps.d_ovps.ps_24 + band * iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR], 1);

  en2 = en2p * tau.get_gfn_tau({0}, band-offBand, false) + en2m * tau.get_gfn_tau({0}, band-offBand, true);

  nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  en2 = en2 * tau.get_wgt(1) / static_cast<double>(nsamp);

  egf2.front() += en2;
}
void QC_monte::mcgf2_local_energy_diff(std::vector<double>& egf2, int band) {
  int ip;
  int nsamp;
  double en2m, en2p;

  en2p = cblas_ddot(iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR],
                    ovps.d_ovps.en2pCore, 1,
                    ovps.d_ovps.ps_24 + band * iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR], 1);
  en2m = cblas_ddot(iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR],
                    ovps.d_ovps.en2mCore, 1,
                    ovps.d_ovps.ps_24 + band * iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR], 1);

  nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  en2p = en2p * tau.get_gfn_tau({0}, band-offBand, false) * tau.get_wgt(1) / static_cast<double>(nsamp);
  en2m = en2m * tau.get_gfn_tau({0}, band-offBand, true) * tau.get_wgt(1) / static_cast<double>(nsamp);

  for (ip = 0; ip < iops.iopns[KEYS::DIFFS]; ip++) {
    if (ip % 2 == 0) {
      egf2[ip] += en2p + en2m;
    } else if (ip % 2 == 1) {
      egf2[ip] += en2p - en2m;
    }
    en2p = en2p * tau.get_tau(0);
    en2m = en2m * tau.get_tau(0);
  }
}
void QC_monte::mcgf2_local_energy_full(int band) {
  int nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  double alpha, beta;

  // ent = alpha * en2p . psi2
  alpha = tau.get_gfn_tau({0}, band-offBand, false) * tau.get_wgt(1) / static_cast<double>(nsamp);
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // ent = alpha * en2m . psi2 + ent
  alpha = tau.get_gfn_tau({0}, band-offBand, true) * tau.get_wgt(1) / static_cast<double>(nsamp);
  beta = 1.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2 = Transpose[psi2] . ent + en2
  alpha = 1.00;
  beta = 1.00;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.en2[band][0], ivir2 - iocc1);
}
void QC_monte::mcgf2_local_energy_full_diff(int band) {
  int nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  double alpha, beta;

  // ent = alpha * en2pCore . psi2
  alpha = tau.get_gfn_tau({0}, band-offBand, false) * tau.get_wgt(1) / static_cast<double>(nsamp);
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2p = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.en2p, ivir2 - iocc1);

  // ent = alpha * en2mCore . psi2
  alpha = tau.get_gfn_tau({0}, band-offBand, true) * tau.get_wgt(1) / static_cast<double>(nsamp);
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2m = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.en2m, ivir2 - iocc1);
}

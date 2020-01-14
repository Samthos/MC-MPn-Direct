#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"

void gf2_core(OVPs& ovps, Electron_Pair_List* electron_pair_list) {
  int tidx, tidy;
  int mc_pair_num = electron_pair_list->size();
#define TIDX_CONTROL for (tidx = 0; tidx < mc_pair_num; tidx++)
#define TIDY_CONTROL for (tidy = 0; tidy < mc_pair_num; tidy++)
  TIDX_CONTROL {
    TIDY_CONTROL {
      int Index = tidy * mc_pair_num + tidx;
      double en2m, en2p;
      en2m = 0;
      en2p = 0;
      if (tidx != tidy) {
        en2p = en2p - 2.00 * ovps.o_set[0][0].s_11[Index] * ovps.v_set[0][0].s_22[Index] * ovps.v_set[0][0].s_11[Index];  // ovps.ps_.s_22[bandIndex];
        en2p = en2p + 1.00 * ovps.o_set[0][0].s_11[Index] * ovps.v_set[0][0].s_12[Index] * ovps.v_set[0][0].s_21[Index];  // ovps.ps_.s_22[bandIndex];

        en2m = en2m + 2.00 * ovps.o_set[0][0].s_22[Index] * ovps.o_set[0][0].s_11[Index] * ovps.v_set[0][0].s_11[Index];  // ovps.ps_.s_22c[bandIndex];
        en2m = en2m - 1.00 * ovps.o_set[0][0].s_21[Index] * ovps.o_set[0][0].s_12[Index] * ovps.v_set[0][0].s_11[Index];  // ovps.ps_.s_22c[bandIndex];

        en2p = en2p * electron_pair_list->rv[tidx] * electron_pair_list->rv[tidy];
        en2m = en2m * electron_pair_list->rv[tidx] * electron_pair_list->rv[tidy];
      }
      ovps.d_ovps.en2pCore[Index] = en2p;
      ovps.d_ovps.en2mCore[Index] = en2m;
    }
  }
#undef TIDX_CONTROL
#undef TIDY_CONTROL
}

void GF::mcgf2_local_energy_core() {
  gf2_core(ovps, electron_pair_list);
}

void GF::mcgf2_local_energy(std::vector<std::vector<double>>& egf2) {
  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp = nsamp * (nsamp - 1.0);
  double en2;
  double alpha, beta;
  const double *psi2;
  for (int band = 0; band < numBand; band++) {
    if (band-offBand < 0) {
      psi2 = electron_pair_psi2.occ() + (band+iocc2-iocc1-offBand);
    } else {
      psi2 = electron_pair_psi2.vir() + (band-offBand);
    }

    // ent = ovps.ovps.tg_val1[band] * en2pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        beta,
        ovps.d_ovps.ent, 1);

    // ent = ovps.ovps.tg_val1[band] * en2mCore . psi + ent
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        beta,
        ovps.d_ovps.ent, 1);

    // en2 = psi2 . ent
    en2 = cblas_ddot(iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        ovps.d_ovps.ent, 1);

    en2 = en2 * tau->get_wgt(1) / nsamp;

    egf2[band].front() += en2;
  }
}

void GF::mcgf2_local_energy_diff(std::vector<std::vector<double>>& egf2) {
  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp = nsamp * (nsamp - 1.0);
  double en2m, en2p;
  double alpha, beta;
  const double *psi2;
  for (int band = 0; band < numBand; band++) {
    if (band-offBand < 0) {
      psi2 = electron_pair_psi2.occ() + (band+iocc2-iocc1-offBand);
    } else {
      psi2 = electron_pair_psi2.vir() + (band-offBand);
    }

    // ent = en2pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        beta,
        ovps.d_ovps.ent, 1);

    // en2p = psi2 . ent
    en2p = cblas_ddot(iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        ovps.d_ovps.ent, 1);

    // ent = en2mCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        beta,
        ovps.d_ovps.ent, 1);

    // en2m = psi2 . ent
    en2m = cblas_ddot(iops.iopns[KEYS::MC_NPAIR],
        psi2, electron_pair_psi2.lda,
        ovps.d_ovps.ent, 1);

    en2p = en2p * tau->get_wgt(1) / nsamp;
    en2m = en2m * tau->get_wgt(1) / nsamp;
    for (int diff=0; diff < iops.iopns[KEYS::DIFFS]; diff++){
      if (diff%2==0) {
        egf2[band][diff] += en2p+en2m;
      } else if (diff%2==1) {
        egf2[band][diff] += en2p-en2m;
      }
      en2p = en2p * tau->get_tau(0);
      en2m = en2m * tau->get_tau(0);
    }
  }
}

void GF::mcgf2_local_energy_full(int band) {
  double nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp = nsamp * (nsamp - 1.0);
  double alpha, beta;

  alpha = tau->get_gfn_tau(0, 0, band-offBand, false) * tau->get_wgt(1) / nsamp;
  beta  = tau->get_gfn_tau(0, 0, band-offBand, true ) * tau->get_wgt(1) / nsamp;
  std::transform(ovps.d_ovps.en2pCore,
      ovps.d_ovps.en2pCore + iops.iopns[KEYS::MC_NPAIR] * iops.iopns[KEYS::MC_NPAIR],
      ovps.d_ovps.en2mCore,
      ovps.d_ovps.enCore,
      [&](double a, double b) {return alpha*a + beta*b;});

  // ent = alpha * en2p . psi2
  alpha = 1.0;
  beta  = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      iops.iopns[KEYS::MC_NPAIR], ivir2-iocc1, iops.iopns[KEYS::MC_NPAIR],
      alpha,
      ovps.d_ovps.enCore, iops.iopns[KEYS::MC_NPAIR],
      electron_pair_psi2.occ(), electron_pair_psi2.lda,
      beta,
      ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2 = Transpose[psi2] . ent + en2
  alpha = 1.00;
  beta  = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::MC_NPAIR],
      alpha,
      electron_pair_psi2.occ(), electron_pair_psi2.lda,
      ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
      beta,
      ovps.d_ovps.enBlock[band][0], ivir2-iocc1);
}

void GF::mcgf2_local_energy_full_diff(int band) {
  double nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp = nsamp * (nsamp - 1.0);
  double alpha, beta;

  // ent = contraction_exp * en2pCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / nsamp;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR],
              alpha,
              ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
              electron_pair_psi2.occ(), electron_pair_psi2.lda,
              beta,
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2p = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR],
              alpha,
              electron_pair_psi2.occ(), electron_pair_psi2.lda,
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
              beta,
              ovps.d_ovps.en2p, ivir2 - iocc1);

  // ent = contraction_exp * en2mCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / nsamp;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR],
              alpha,
              ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
              electron_pair_psi2.occ(), electron_pair_psi2.lda,
              beta,
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // en2m = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR],
              alpha,
              electron_pair_psi2.occ(), electron_pair_psi2.lda,
              ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR],
              beta,
              ovps.d_ovps.en2m, ivir2 - iocc1);
}

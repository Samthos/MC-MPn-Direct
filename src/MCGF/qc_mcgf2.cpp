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

        en2p = en2p * electron_pair_list->get(tidx).rv * electron_pair_list->get(tidy).rv;
        en2m = en2m * electron_pair_list->get(tidx).rv * electron_pair_list->get(tidy).rv;
      }
      ovps.d_ovps.en2pCore[Index] = en2p;
      ovps.d_ovps.en2mCore[Index] = en2m;
    }
  }
#undef TIDX_CONTROL
#undef TIDY_CONTROL
}

void GF::mcgf2_local_energy_core() {
  gf2_core(ovps, el_pair_list);
}

void GF::mcgf2_local_energy(std::vector<std::vector<double>>& egf2) {
  double nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp = nsamp * (nsamp - 1.0);
  double en2;
  double alpha, beta;
  double *psi2;
  for (int band = 0; band < numBand; band++) {
    if (band-offBand < 0) {
      psi2 = basis.h_basis.occ2 + (band+iocc2-iocc1-offBand);
    } else {
      psi2 = basis.h_basis.vir2 + (band-offBand);
    }

    // ent = ovps.ovps.tg_val1[band] * en2pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, ivir2 - iocc1,
        beta,
        ovps.d_ovps.ent, 1);

    // ent = ovps.ovps.tg_val1[band] * en2mCore . psi + ent
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::MC_NPAIR],
        alpha,
        ovps.d_ovps.en2mCore, iops.iopns[KEYS::MC_NPAIR],
        psi2, ivir2 - iocc1,
        beta,
        ovps.d_ovps.ent, 1);

    // en2 = psi2 . ent
    en2 = cblas_ddot(iops.iopns[KEYS::MC_NPAIR],
        psi2, ivir2 - iocc1,
        ovps.d_ovps.ent, 1);

    en2 = en2 * tau->get_wgt(1) / nsamp;

    egf2[band].front() += en2;
  }

}
void GF::mcgf2_local_energy_diff(std::vector<double>& egf2, int band) {
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
  en2p = en2p * tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / static_cast<double>(nsamp);
  en2m = en2m * tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / static_cast<double>(nsamp);

  for (ip = 0; ip < iops.iopns[KEYS::DIFFS]; ip++) {
    if (ip % 2 == 0) {
      egf2[ip] += en2p + en2m;
    } else if (ip % 2 == 1) {
      egf2[ip] += en2p - en2m;
    }
    en2p = en2p * tau->get_tau(0);
    en2m = en2m * tau->get_tau(0);
  }
}
void GF::mcgf2_local_energy_full(int band) {
  int nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  double alpha, beta;

  // ent = contraction_exp * en2p . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / static_cast<double>(nsamp);
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              iops.iopns[KEYS::MC_NPAIR], ivir2 - iocc1, iops.iopns[KEYS::MC_NPAIR], alpha,
              ovps.d_ovps.en2pCore, iops.iopns[KEYS::MC_NPAIR],
              ovps.d_ovps.psi2, iops.iopns[KEYS::MC_NPAIR],
              beta, ovps.d_ovps.ent, iops.iopns[KEYS::MC_NPAIR]);

  // ent = contraction_exp * en2m . psi2 + ent
  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / static_cast<double>(nsamp);
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
void GF::mcgf2_local_energy_full_diff(int band) {
  int nsamp = iops.iopns[KEYS::MC_NPAIR] * (iops.iopns[KEYS::MC_NPAIR] - 1);
  double alpha, beta;

  // ent = contraction_exp * en2pCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / static_cast<double>(nsamp);
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

  // ent = contraction_exp * en2mCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / static_cast<double>(nsamp);
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

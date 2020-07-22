#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"
#include "qc_mcgf2.h"

void gf2_core(OVPS_Host& ovps, OVPS_ARRAY& d_ovps, Electron_Pair_List* electron_pair_list) {
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
      d_ovps.en2pCore[Index] = en2p;
      d_ovps.en2mCore[Index] = en2m;
    }
  }
#undef TIDX_CONTROL
#undef TIDY_CONTROL
}

void GF::mcgf2_local_energy_core() {
  gf2_core(ovps, d_ovps, electron_pair_list);
}

void GF::mcgf2_local_energy_full(int band) {
  double nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0);
  double alpha, beta;

  alpha = tau->get_gfn_tau(0, 0, band-offBand, false) * tau->get_wgt(1) / nsamp;
  beta  = tau->get_gfn_tau(0, 0, band-offBand, true ) * tau->get_wgt(1) / nsamp;
  std::transform(d_ovps.en2pCore.begin(),
      d_ovps.en2pCore.end(),
      d_ovps.en2mCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + beta*b;});

  // ent = alpha * en2p . psi2
  alpha = 1.0;
  beta  = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
      alpha,
      d_ovps.enCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      beta,
      d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS]);

  // en2 = Transpose[psi2] . ent + en2
  alpha = 1.00;
  beta  = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
      alpha,
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
      beta,
      d_ovps.enBlock[band][0].data(), ivir2-iocc1);
}

void GF::mcgf2_local_energy_full_diff(int band) {
  double nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0);
  double alpha, beta;

  // ent = contraction_exp * en2pCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / nsamp;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              iops.iopns[KEYS::ELECTRON_PAIRS], ivir2 - iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
              alpha,
              d_ovps.en2pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
              wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
              beta,
              d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS]);

  // en2p = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
              alpha,
              wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
              d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
              beta,
              d_ovps.en2p, ivir2 - iocc1);

  // ent = contraction_exp * en2mCore . psi2
  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / nsamp;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              iops.iopns[KEYS::ELECTRON_PAIRS], ivir2 - iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
              alpha,
              d_ovps.en2mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
              wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
              beta,
              d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS]);

  // en2m = Transpose[psi2] . ent
  alpha = 1.00;
  beta = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ivir2 - iocc1, ivir2 - iocc1, iops.iopns[KEYS::ELECTRON_PAIRS],
              alpha,
              wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
              d_ovps.ent.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
              beta,
              d_ovps.en2m, ivir2 - iocc1);
}

GF2_Functional::GF2_Functional(IOPs& iops) :
  MCGF(iops, 1, "22", false),
  en2mCore(n_electron_pairs * n_electron_pairs),
  en2pCore(n_electron_pairs * n_electron_pairs)
{
  ent.resize(n_electron_pairs);
  nsamp = static_cast<double>(n_electron_pairs);
  nsamp = nsamp * (nsamp - 1.0);


  if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL || 
        iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
    std::cerr << "Full rountine not integrated into MCGF class\n";
    exit(0);
    // ent.resize((basis.ivir2 - basis.iocc1) * n_electron_pairs);
  }
}

void GF2_Functional::core(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  int tidx, tidy;
  for (tidx = 0; tidx < n_electron_pairs; tidx++) {
    for (tidy = 0; tidy < n_electron_pairs; tidy++) {
      int Index = tidy * n_electron_pairs + tidx;
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
      en2pCore[Index] = en2p;
      en2mCore[Index] = en2m;
    }
  }
}

void GF2_Functional::energy_no_diff(std::vector<std::vector<double>>& egf2,
       std::unordered_map<int, Wavefunction>& wavefunctions,
       Electron_Pair_List* electron_pair_list, Tau* tau) {
  double en2;
  double alpha, beta;
  const double *psi2;
  for (int band = 0; band < numBand; band++) {
    //if (band-offBand < 0) {
    //  psi2 = wavefunctions[WC::electron_pairs_2].occ() + (band+iocc2-iocc1-offBand);
    //} else {
    //}
    psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);

    // ent = _val1[band] * en2pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en2pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = _val1[band] * en2mCore . psi + ent
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en2mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // en2 = psi2 . ent
    en2 = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);

    en2 = en2 * tau->get_wgt(1) / nsamp;

    egf2[band].front() += en2;
  }
}

void GF2_Functional::energy_diff(std::vector<std::vector<double>>& egf2,
       std::unordered_map<int, Wavefunction>& wavefunctions,
       Electron_Pair_List* electron_pair_list, Tau* tau
    ) {
  double en2m, en2p;
  double alpha, beta;
  const double *psi2;
  for (int band = 0; band < numBand; band++) {
    //if (band-offBand < 0) {
    //  psi2 = wavefunctions[WC::electron_pairs_2].occ() + (band+iocc2-iocc1-offBand);
    //} else {
    //  psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);
    //}
    psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);

    // ent = en2pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en2pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // en2p = psi2 . ent
    en2p = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);

    // ent = en2mCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en2mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // en2m = psi2 . ent
    en2m = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);

    en2p = en2p * tau->get_wgt(1) / nsamp;
    en2m = en2m * tau->get_wgt(1) / nsamp;
    for (int diff=0; diff < numDiff; diff++){
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

void GF2_Functional::energy_f12(std::vector<std::vector<double>>&, 
   std::unordered_map<int, Wavefunction>&,
   Electron_Pair_List*, Electron_List*) {}

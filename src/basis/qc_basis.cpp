// Copyright 2017

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#ifdef USE_MPI
#include "mpi.h"
#endif
#include "qc_basis.h"
#include "../atom_znum.h"

Basis::Basis() {
  cf[0] = sqrt(2.5) * 0.5;
  cf[1] = sqrt(2.5) * 1.5;
  cf[2] = sqrt(15.0);
  cf[3] = sqrt(1.5) * 0.5;
  cf[4] = sqrt(6.0);
  cf[5] = 1.5;
  cf[6] = sqrt(15.0) * 0.5;

  cg[0] = 2.9580398915498085;  // (3, 1, 0) (1, 3, 0)
  cg[1] = 6.2749501990055672;
  cg[2] = 2.0916500663351894;
  cg[3] = 1.1180339887498949;
  cg[4] = 6.7082039324993694;
  cg[5] = 2.3717082451262845;
  cg[6] = 3.1622776601683795;
  cg[7] = 0.55901699437494745;
  cg[8] = 3.3541019662496847;
  cg[9] = 0.73950997288745213;
  cg[10] = 4.4370598373247132;
}
Basis::~Basis() {
  delete[] nw_en;
  delete[] h_basis.nw_co;

  delete[] h_basis.icgs;
  delete[] h_basis.alpha;
  delete[] h_basis.norm;
  delete[] h_basis.am;
  delete[] h_basis.at;
  delete[] h_basis.stop_list;
  delete[] h_basis.isgs;
}
Basis::Basis(const Basis& param) {
  qc_nprm = param.qc_nprm;
  qc_ncgs = param.qc_ncgs;
  qc_ngfs = param.qc_ngfs;
  qc_nshl = param.qc_nshl;
  lspherical = param.lspherical;

  h_basis.am = new int[qc_nshl];
  h_basis.at = new int[qc_nshl];
  h_basis.isgs = new int[qc_nshl + 1];
  h_basis.stop_list = new int[qc_nshl + 1];

  std::copy(param.h_basis.am, param.h_basis.am + qc_nshl, h_basis.am);
  std::copy(param.h_basis.at, param.h_basis.at + qc_nshl, h_basis.at);
  std::copy(param.h_basis.isgs, param.h_basis.isgs + qc_nshl + 1, h_basis.isgs);
  std::copy(param.h_basis.stop_list, param.h_basis.stop_list + qc_nshl + 1, h_basis.stop_list);

  iocc1 = param.iocc1;
  iocc2 = param.iocc2;
  ivir1 = param.ivir1;
  ivir2 = param.ivir2;

  // from nwchem
  nw_nsets = param.nw_nsets;
  nw_nbf = param.nw_nbf;
  nw_iocc = param.nw_iocc;
  nw_icore = param.nw_icore;
  nw_nmo[0] = param.nw_nmo[0];
  nw_nmo[1] = param.nw_nmo[1];

  h_basis.icgs = new double[nw_nbf];
  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo[0]];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nmo[0] * nw_nbf, h_basis.nw_co);

  h_basis.alpha = new double[qc_nprm];
  h_basis.norm = new double[qc_nprm];
  std::copy(param.h_basis.alpha, param.h_basis.alpha + qc_nprm, h_basis.alpha);
  std::copy(param.h_basis.norm, param.h_basis.norm + qc_nprm, h_basis.norm);
}
Basis& Basis::operator=(const Basis& param) {
  qc_nprm = param.qc_nprm;
  qc_ncgs = param.qc_ncgs;
  qc_ngfs = param.qc_ngfs;
  qc_nshl = param.qc_nshl;
  lspherical = param.lspherical;

  h_basis.am = new int[qc_nshl];
  h_basis.at = new int[qc_nshl];
  h_basis.isgs = new int[qc_nshl + 1];
  h_basis.stop_list = new int[qc_nshl + 1];

  std::copy(param.h_basis.am, param.h_basis.am + qc_nshl, h_basis.am);
  std::copy(param.h_basis.at, param.h_basis.at + qc_nshl, h_basis.at);
  std::copy(param.h_basis.isgs, param.h_basis.isgs + qc_nshl + 1, h_basis.isgs);
  std::copy(param.h_basis.stop_list, param.h_basis.stop_list + qc_nshl + 1, h_basis.stop_list);

  iocc1 = param.iocc1;
  iocc2 = param.iocc2;
  ivir1 = param.ivir1;
  ivir2 = param.ivir2;

  // from nwchem
  nw_nsets = param.nw_nsets;
  nw_nbf = param.nw_nbf;
  nw_iocc = param.nw_iocc;
  nw_icore = param.nw_icore;
  nw_nmo[0] = param.nw_nmo[0];
  nw_nmo[1] = param.nw_nmo[1];

  h_basis.icgs = new double[nw_nbf];
  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo[0]];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nbf * nw_nmo[0], h_basis.nw_co);

  h_basis.alpha = new double[qc_nprm];
  h_basis.norm = new double[qc_nprm];
  std::copy(param.h_basis.alpha, param.h_basis.alpha + qc_nprm, h_basis.alpha);
  std::copy(param.h_basis.norm, param.h_basis.norm + qc_nprm, h_basis.norm);

  return *this;
}

void Basis::read(IOPs& iops, MPI_info& mpi_info, Molec& molec) {
  std::ifstream input;

  int i, j, k;
  int znum, nshell;
  int ncgs0, nsgs0, nprm0;
  int ncgs, nsgs, nshl, nprm;
  int nprim;
  std::string atname, sym;

  if (mpi_info.sys_master) {
    std::cout << "Basis set: " << iops.sopns[KEYS::BASIS] << std::endl;
    input.open(iops.sopns[KEYS::BASIS].c_str());
    if (input.is_open()) {
      // Gaussian94 format
      // S, SP, P, D
      nprm = 0;
      ncgs = 0;
      nsgs = 0;
      nshl = 0;
      while (input.peek() >= 65 && input.peek() <= 90) {  // i.e. while next character is a captial letter
        input >> atname >> nshell;
        znum = atomic_znum(atname);

        ncgs0 = 0;
        nsgs0 = 0;
        nprm0 = 0;
        for (i = 0; i < nshell; i++) {
          input >> sym >> nprim;
          input.ignore(256, '\n');
          for (j = 0; j < nprim; j++) {
            input.ignore(256, '\n');
          }

          if (sym == "S") {
            ncgs0 = ncgs0 + 1;
            nsgs0 = nsgs0 + 1;
          } else if (sym == "SP") {
            ncgs0 = ncgs0 + 4;
            nsgs0 = nsgs0 + 4;
          } else if (sym == "P") {
            ncgs0 = ncgs0 + 3;
            nsgs0 = nsgs0 + 3;
          } else if (sym == "D") {
            ncgs0 = ncgs0 + 6;
            nsgs0 = nsgs0 + 5;
          } else if (sym == "F") {
            ncgs0 = ncgs0 + 10;
            nsgs0 = nsgs0 + 7;
          } else if (sym == "G") {
            ncgs0 = ncgs0 + 15;
            nsgs0 = nsgs0 + 9;
          }
          nprm0 = nprm0 + nprim;
        }

        for (i = 0; i < molec.natom; i++) {
          if (znum == molec.atom[i].znum) {
            nshl = nshl + nshell;
            ncgs = ncgs + ncgs0;
            nsgs = nsgs + nsgs0;
            nprm = nprm + nprm0;
          }
        }
      }
    } else {
      std::cerr << "No basis file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&nshl, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nsgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nprm, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  qc_nprm = nprm;
  qc_ncgs = ncgs;
  qc_nshl = nshl;

  if (iops.bopns[KEYS::SPHERICAL]) {
    lspherical = true;
    qc_ngfs = nsgs;
  } else {
    lspherical = false;
    qc_ngfs = ncgs;
  }

  h_basis.alpha = new double[qc_nprm];
  h_basis.norm = new double[qc_nprm];
  h_basis.am = new int[qc_nshl];
  h_basis.at = new int[qc_nshl];
  h_basis.stop_list = new int[qc_nshl + 1];
  h_basis.isgs = new int[qc_nshl + 1];

  nprm = 0;
  nshl = 0;
  ncgs = 0;
  nsgs = 0;
  int new_sgs = 0;
  int kk = 0;

  h_basis.stop_list[0] = 0;
  h_basis.isgs[0] = 0;

  for (i = 0; i < molec.natom; i++) {
    if (mpi_info.sys_master) {
      input.clear();
      input.seekg(0, std::ios::beg);

      while (input.peek() >= 65 && input.peek() <= 90) {  // i.e. while next character is a captial letter
        input >> atname >> nshell;
        znum = atomic_znum(atname);

        if (znum == molec.atom[i].znum) {
          for (j = 0; j < nshell; j++) {
            input >> sym >> nprim;
            input.ignore(256, '\n');
            if (sym == "SP") {
              new_sgs = new_sgs + 4;
            } else if (sym == "S") {
              h_basis.am[kk] = 0;
              new_sgs = new_sgs + 1;
            } else if (sym == "P") {
              h_basis.am[kk] = 1;
              new_sgs = new_sgs + 3;
            } else if (sym == "D") {
              h_basis.am[kk] = 2;
              new_sgs = new_sgs + 5;
            } else if (sym == "F") {
              h_basis.am[kk] = 3;
              new_sgs = new_sgs + 7;
            } else if (sym == "G") {
              h_basis.am[kk] = 4;
              new_sgs = new_sgs + 9;
            }

            if (h_basis.am[kk] == -1) {
              for (k = 0; k < nprim; k++) {
                // input >> alpha[k][j] >> coef[k][j] >> coef2[k][j];
              }
            } else {
              for (k = 0; k < nprim; k++) {
                input >> h_basis.alpha[nprm] >> h_basis.norm[nprm];
                nprm++;
              }
            }
            h_basis.at[kk] = i;
            h_basis.stop_list[kk + 1] = nprm;
            h_basis.isgs[kk + 1] = new_sgs;
            kk++;
          }
        } else {
          for (j = 0; j < nshell; j++) {
            input >> sym >> nprim;
            input.ignore(1000, '\n');
            for (k = 0; k < nprim; k++) {
              input.ignore(1000, '\n');
            }
          }
        }
      }
    }
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(h_basis.alpha, qc_nprm, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.norm, qc_nprm, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.am, qc_nshl, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.at, qc_nshl, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.stop_list, qc_nshl + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.isgs, qc_nshl + 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  normalize();

  /*
  for (i = 0; i < qc_nprm; i++) {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (mpi_info.taskid == 0) {
        printf("%2i\t%7.3f\t%7.3f\n", i, h_basis.alpha[i], h_basis.norm[i]);
        fflush(stdout);
      }
  }
  for (i = 0; i < qc_nprm; i++) {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (mpi_info.taskid == 1) {
        printf("%2i\t%7.3f\t%7.3f\n", i, h_basis.alpha[i], h_basis.norm[i]);
        fflush(stdout);
      }
  }
*/

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // if (mpi_info.taskid == 0) {
  //   std::cout << "NSHL\t" << qc_nshl << "\t" << mpi_info.taskid << std::endl;
  //   std::cout << "NGFS\t" << qc_ngfs << "\t" << mpi_info.taskid << std::endl;
  //   std::cout << "NCGS\t" << qc_ncgs << "\t" << mpi_info.taskid << std::endl;
  // }
}

void Basis::normalize() {
  int i, j, k;
  double cnorm, aa, dum, fac, facs, pi32;
  constexpr double pi = 3.141592653589793;

  for (i = 0; i < qc_nshl; i++) {  // number of shells on the atom
    if (h_basis.am[i] == -1) {
      /*
      qc_shl_list[nshl[0]].ncgs = 4;
      qc_shl_list[nshl[0]].nsgs = 4;

      qc_shl_list[nshl[0]].h_basis.norm = new double*[2];
      for (j = 0; j < 2; j++) {
        qc_shl_list[nshl[0]].h_basis.norm[j] = new double[nprim[i]];
      }

      nsgs = nsgs + 4;
      ncgs = ncgs + 4;

      for (j = 0; j < nprim[i]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j][i] / pi));
        qc_shl_list[nshl[0]].h_basis.alpha[j]  = h_basis.alpha[j][i];
        qc_shl_list[nshl[0]].h_basis.norm[1][j] = coef[j][i] * ch_basis.norm;
        cnorm = cnorm * sqrt(4.0 * new_alpha[j][i]);
        qc_shl_list[nshl[0]].new_norm[2][j]  = coef2[j][i] * cnorm;
      }
*/
    } else if (h_basis.am[i] == 0) {
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j] / pi));
        h_basis.norm[j] = h_basis.norm[j] * cnorm;
      }

      // ncgs = ncgs + 1;
      // nsgs[0] = nsgs[0] + 1;
      facs = 0.0;
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        for (k = h_basis.stop_list[i]; k <= j; k++) {
          aa = h_basis.alpha[j] + h_basis.alpha[k];
          fac = aa * sqrt(aa);
          dum = h_basis.norm[j] * h_basis.norm[k] / fac;
          if (j != k) {
            dum = dum + dum;
          }
          facs = facs + dum;
        }
      }
      pi32 = 5.56832799683170;
      facs = 1.0 / sqrt(facs * pi32);

      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        h_basis.norm[j] = h_basis.norm[j] * facs;
      }
    } else if (h_basis.am[i] == 1) {
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j] / pi));
        cnorm = cnorm * sqrt(4.0 * h_basis.alpha[j]);
        h_basis.norm[j] = h_basis.norm[j] * cnorm;
      }
      // ncgs = ncgs + 3;
      // nsgs[0] = nsgs[0] + 3;

      facs = 0.0;
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        for (k = h_basis.stop_list[i]; k <= j; k++) {
          aa = h_basis.alpha[j] + h_basis.alpha[k];
          fac = aa * sqrt(aa);
          dum = 0.5 * h_basis.norm[j] * h_basis.norm[k] / (aa * fac);
          if (j != k) {
            dum = dum + dum;
          }
          facs = facs + dum;
        }
      }
      pi32 = 5.56832799683170;
      facs = 1.0 / sqrt(facs * pi32);

      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        h_basis.norm[j] = h_basis.norm[j] * facs;
      }
    } else if (h_basis.am[i] == 2) {
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j] / pi)) * 4.0 * h_basis.alpha[j];
        cnorm = cnorm / sqrt(3.0);
        h_basis.norm[j] = h_basis.norm[j] * cnorm;  // dxx
      }
      // ncgs = ncgs + 6;
      // nsgs[0] = nsgs[0] + 5;
    } else if (h_basis.am[i] == 3) {
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j] / pi)) * pow(4.0 * h_basis.alpha[j], 1.5);
        cnorm = cnorm / sqrt(15.0);
        h_basis.norm[j] = h_basis.norm[j] * cnorm;  // dxx
      }

      // ncgs = ncgs + 10;
      // nsgs[0] = nsgs[0] + 7;
    } else if (h_basis.am[i] == 4) {
      for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j] / pi)) * pow(4.0 * h_basis.alpha[j], 2.0);
        cnorm = cnorm / sqrt(7.0 * 15.0);
        h_basis.norm[j] = h_basis.norm[j] * cnorm;  // dxx
      }
      // ncgs = ncgs + 15;
      // nsgs[0] = nsgs[0] + 9;
    }
    // nshl[0] = nshl[0] + 1;
  }
}

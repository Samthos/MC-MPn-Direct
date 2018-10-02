// Copyright 2017

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#ifdef HAVE_MPI
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
  delete[] h_basis.ao_amplitudes;
  delete[] h_basis.contraction_exp;
  delete[] h_basis.contraction_coef;
}
Basis::Basis(const Basis& param) {
  qc_nprm = param.qc_nprm;
  qc_ncgs = param.qc_ncgs;
  qc_ngfs = param.qc_ngfs;
  qc_nshl = param.qc_nshl;
  lspherical = param.lspherical;

  h_basis.meta_data = new BasisMetaData[qc_nshl];
  std::copy(param.h_basis.meta_data, param.h_basis.meta_data + qc_nshl, h_basis.meta_data);

  iocc1 = param.iocc1;
  iocc2 = param.iocc2;
  ivir1 = param.ivir1;
  ivir2 = param.ivir2;

  // from nwchem
  nw_nbf = param.nw_nbf;
  nw_nmo = param.nw_nmo;

  h_basis.ao_amplitudes = new double[nw_nbf];
  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nmo * nw_nbf, h_basis.nw_co);

  h_basis.contraction_exp = new double[qc_nprm];
  h_basis.contraction_coef = new double[qc_nprm];
  std::copy(param.h_basis.contraction_exp, param.h_basis.contraction_exp + qc_nprm, h_basis.contraction_exp);
  std::copy(param.h_basis.contraction_coef, param.h_basis.contraction_coef + qc_nprm, h_basis.contraction_coef);
}
Basis& Basis::operator=(const Basis& param) {
  qc_nprm = param.qc_nprm;
  qc_ncgs = param.qc_ncgs;
  qc_ngfs = param.qc_ngfs;
  qc_nshl = param.qc_nshl;
  lspherical = param.lspherical;

  h_basis.meta_data = new BasisMetaData[qc_nshl];
  std::copy(param.h_basis.meta_data, param.h_basis.meta_data + qc_nshl, h_basis.meta_data);

  iocc1 = param.iocc1;
  iocc2 = param.iocc2;
  ivir1 = param.ivir1;
  ivir2 = param.ivir2;

  // from nwchem
  nw_nbf = param.nw_nbf;
  nw_nmo = param.nw_nmo;

  h_basis.ao_amplitudes = new double[nw_nbf];
  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nbf * nw_nmo, h_basis.nw_co);

  h_basis.contraction_exp = new double[qc_nprm];
  h_basis.contraction_coef = new double[qc_nprm];
  std::copy(param.h_basis.contraction_exp, param.h_basis.contraction_exp + qc_nprm, h_basis.contraction_exp);
  std::copy(param.h_basis.contraction_coef, param.h_basis.contraction_coef + qc_nprm, h_basis.contraction_coef);

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

#ifdef HAVE_MPI
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

  h_basis.contraction_exp = new double[qc_nprm];
  h_basis.contraction_coef = new double[qc_nprm];
  h_basis.meta_data = new BasisMetaData[qc_nshl];

  nprm = 0;
  nshl = 0;
  ncgs = 0;
  nsgs = 0;

  int shell = 0;
  int contraction_begin = 0;
  int ao_begin = 0;
  int ao_offset = 0;

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
              ao_offset = 4;
            } else if (sym == "S") {
              h_basis.meta_data[shell].angular_moment = 0;
              ao_offset = 1;
            } else if (sym == "P") {
              h_basis.meta_data[shell].angular_moment = 1;
              ao_offset = 3;
            } else if (sym == "D") {
              h_basis.meta_data[shell].angular_moment = 2;
              ao_offset = 5;
            } else if (sym == "F") {
              h_basis.meta_data[shell].angular_moment = 3;
              ao_offset = 7;
            } else if (sym == "G") {
              h_basis.meta_data[shell].angular_moment = 4;
              ao_offset = 9;
            }

            if (h_basis.meta_data[shell].angular_moment == -1) {
              for (k = 0; k < nprim; k++) {
                // input >> contraction_exp[k][j] >> coef[k][j] >> coef2[k][j];
              }
            } else {
              for (k = 0; k < nprim; k++) {
                input >> h_basis.contraction_exp[contraction_begin + k];
                input >> h_basis.contraction_coef[contraction_begin + k];
              }
            }
            std::copy(molec.atom[i].pos, molec.atom[i].pos + 3, h_basis.meta_data[shell].pos);

            h_basis.meta_data[shell].contraction_begin = contraction_begin;
            contraction_begin +=nprim;
            h_basis.meta_data[shell].contraction_end = contraction_begin;

            h_basis.meta_data[shell].ao_begin = ao_begin;
            ao_begin += ao_offset;

            shell++;
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

#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(h_basis.contraction_exp, qc_nprm, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.contraction_coef, qc_nprm, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.meta_data, qc_nshl * sizeof(BasisMetaData), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  normalize();

  /*
  for (i = 0; i < qc_nprm; i++) {
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (mpi_info.taskid == 0) {
        printf("%2i\t%7.3f\t%7.3f\n", i, h_basis.contraction_exp[i], h_basis.contraction_coef[i]);
        fflush(stdout);
      }
  }
  for (i = 0; i < qc_nprm; i++) {
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (mpi_info.taskid == 1) {
        printf("%2i\t%7.3f\t%7.3f\n", i, h_basis.contraction_exp[i], h_basis.contraction_coef[i]);
        fflush(stdout);
      }
  }
*/

#ifdef HAVE_MPI
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
    if (h_basis.meta_data[i].angular_moment == -1) {
      /*
      qc_shl_list[nshl[0]].ncgs = 4;
      qc_shl_list[nshl[0]].nsgs = 4;

      qc_shl_list[nshl[0]].h_basis.contraction_coef = new double*[2];
      for (j = 0; j < 2; j++) {
        qc_shl_list[nshl[0]].h_basis.contraction_coef[j] = new double[nprim[i]];
      }

      nsgs = nsgs + 4;
      ncgs = ncgs + 4;

      for (j = 0; j < nprim[i]; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j][i] / pi));
        qc_shl_list[nshl[0]].h_basis.contraction_exp[j]  = h_basis.contraction_exp[j][i];
        qc_shl_list[nshl[0]].h_basis.contraction_coef[1][j] = coef[j][i] * ch_basis.contraction_coef;
        cnorm = cnorm * sqrt(4.0 * new_alpha[j][i]);
        qc_shl_list[nshl[0]].new_norm[2][j]  = coef2[j][i] * cnorm;
      }
*/
    } else if (h_basis.meta_data[i].angular_moment == 0) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi));
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;
      }

      // ncgs = ncgs + 1;
      // nsgs[0] = nsgs[0] + 1;
      facs = 0.0;
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        for (k = h_basis.meta_data[i].contraction_begin; k <= j; k++) {
          aa = h_basis.contraction_exp[j] + h_basis.contraction_exp[k];
          fac = aa * sqrt(aa);
          dum = h_basis.contraction_coef[j] * h_basis.contraction_coef[k] / fac;
          if (j != k) {
            dum = dum + dum;
          }
          facs = facs + dum;
        }
      }
      pi32 = 5.56832799683170;
      facs = 1.0 / sqrt(facs * pi32);

      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * facs;
      }
    } else if (h_basis.meta_data[i].angular_moment == 1) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi));
        cnorm = cnorm * sqrt(4.0 * h_basis.contraction_exp[j]);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;
      }
      // ncgs = ncgs + 3;
      // nsgs[0] = nsgs[0] + 3;

      facs = 0.0;
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        for (k = h_basis.meta_data[i].contraction_begin; k <= j; k++) {
          aa = h_basis.contraction_exp[j] + h_basis.contraction_exp[k];
          fac = aa * sqrt(aa);
          dum = 0.5 * h_basis.contraction_coef[j] * h_basis.contraction_coef[k] / (aa * fac);
          if (j != k) {
            dum = dum + dum;
          }
          facs = facs + dum;
        }
      }
      pi32 = 5.56832799683170;
      facs = 1.0 / sqrt(facs * pi32);

      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * facs;
      }
    } else if (h_basis.meta_data[i].angular_moment == 2) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi)) * 4.0 * h_basis.contraction_exp[j];
        cnorm = cnorm / sqrt(3.0);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;  // dxx
      }
      // ncgs = ncgs + 6;
      // nsgs[0] = nsgs[0] + 5;
    } else if (h_basis.meta_data[i].angular_moment == 3) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi)) * pow(4.0 * h_basis.contraction_exp[j], 1.5);
        cnorm = cnorm / sqrt(15.0);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;  // dxx
      }

      // ncgs = ncgs + 10;
      // nsgs[0] = nsgs[0] + 7;
    } else if (h_basis.meta_data[i].angular_moment == 4) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi)) * pow(4.0 * h_basis.contraction_exp[j], 2.0);
        cnorm = cnorm / sqrt(7.0 * 15.0);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;  // dxx
      }
      // ncgs = ncgs + 15;
      // nsgs[0] = nsgs[0] + 9;
    }
    // nshl[0] = nshl[0] + 1;
  }
}

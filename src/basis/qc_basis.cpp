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

Basis::Basis(IOPs &iops, MPI_info &mpi_info, Molec &molec) {
  read(iops, mpi_info, molec);
  nw_vectors_read(iops, mpi_info, molec);

  mc_pair_num = iops.iopns[KEYS::MC_NPAIR];
  h_basis.ao_amplitudes = new double[nw_nbf * mc_pair_num];
  h_basis.psi1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psi2 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau2 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.occ1 = h_basis.psi1;
  h_basis.occ2 = h_basis.psi2;
  h_basis.vir1 = h_basis.psi1 + (ivir1-iocc1);
  h_basis.vir2 = h_basis.psi2 + (ivir1-iocc1);
  h_basis.occTau1 = h_basis.psiTau1;
  h_basis.occTau2 = h_basis.psiTau2;
  h_basis.virTau1 = h_basis.psiTau1 + (ivir1-iocc1);
  h_basis.virTau2 = h_basis.psiTau2 + (ivir1-iocc1);
}
Basis::~Basis() {
  delete[] nw_en;

  delete[] h_basis.nw_co;
  delete[] h_basis.ao_amplitudes;
  delete[] h_basis.contraction_exp;
  delete[] h_basis.contraction_coef;

  delete[] h_basis.psi1;
  delete[] h_basis.psi2;
  delete[] h_basis.psiTau1;
  delete[] h_basis.psiTau2;
  h_basis.occ1 = nullptr;
  h_basis.occ2 = nullptr;
  h_basis.vir1 = nullptr;
  h_basis.vir2 = nullptr;
  h_basis.occTau1 = nullptr;
  h_basis.occTau2 = nullptr;
  h_basis.virTau1 = nullptr;
  h_basis.virTau2 = nullptr;
}
Basis::Basis(const Basis& param) {
  nPrimatives = param.nPrimatives;
  qc_nbf = param.qc_nbf;
  nShells = param.nShells;
  lspherical = param.lspherical;

  h_basis.meta_data = new BasisMetaData[nShells];
  std::copy(param.h_basis.meta_data, param.h_basis.meta_data + nShells, h_basis.meta_data);

  iocc1 = param.iocc1;
  iocc2 = param.iocc2;
  ivir1 = param.ivir1;
  ivir2 = param.ivir2;

  // from nwchem
  nw_nbf = param.nw_nbf;
  nw_nmo = param.nw_nmo;

  mc_pair_num = param.mc_pair_num;
  h_basis.ao_amplitudes = new double[nw_nbf * mc_pair_num];
  h_basis.psi1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psi2 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau2 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.occ1 = h_basis.psi1;
  h_basis.occ2 = h_basis.psi2;
  h_basis.vir1 = h_basis.psi1 + (ivir1-iocc1);
  h_basis.vir2 = h_basis.psi2 + (ivir1-iocc1);
  h_basis.occTau1 = h_basis.psiTau1;
  h_basis.occTau2 = h_basis.psiTau2;
  h_basis.virTau1 = h_basis.psiTau1 + (ivir1-iocc1);
  h_basis.virTau2 = h_basis.psiTau2 + (ivir1-iocc1);

  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nmo * nw_nbf, h_basis.nw_co);

  h_basis.contraction_exp = new double[nPrimatives];
  h_basis.contraction_coef = new double[nPrimatives];
  std::copy(param.h_basis.contraction_exp, param.h_basis.contraction_exp + nPrimatives, h_basis.contraction_exp);
  std::copy(param.h_basis.contraction_coef, param.h_basis.contraction_coef + nPrimatives, h_basis.contraction_coef);
}
Basis& Basis::operator=(Basis param) {
  std::swap(*this, param);
  return *this;
}
void swap(Basis& a, Basis& b) {
  std::swap(a.nPrimatives, b.nPrimatives);
  std::swap(a.qc_nbf, b.qc_nbf);
  std::swap(a.nShells, b.nShells);
  std::swap(a.lspherical, b.lspherical);
  std::swap(a.h_basis.meta_data, b.h_basis.meta_data);
  std::swap(a.iocc1, b.iocc1);
  std::swap(a.iocc2, b.iocc2);
  std::swap(a.ivir1, b.ivir1);
  std::swap(a.ivir2, b.ivir2);

  std::swap(a.nw_nbf, b.nw_nbf);
  std::swap(a.nw_nmo, b.nw_nmo);

  std::swap(a.h_basis.ao_amplitudes, b.h_basis.ao_amplitudes);
  std::swap(a.nw_en, b.nw_en);
  std::swap(a.h_basis.nw_co, b.h_basis.nw_co);
  std::swap(a.h_basis.contraction_exp, b.h_basis.contraction_exp);
  std::swap(a.h_basis.contraction_coef, b.h_basis.contraction_coef);

  std::swap(a.h_basis.psi1, b.h_basis.psi1);
  std::swap(a.h_basis.psi2, b.h_basis.psi2);
  std::swap(a.h_basis.psiTau1, b.h_basis.psiTau1);
  std::swap(a.h_basis.psiTau2, b.h_basis.psiTau2);
  std::swap(a.h_basis.occ1, b.h_basis.occ1);
  std::swap(a.h_basis.occ2, b.h_basis.occ2);
  std::swap(a.h_basis.vir1, b.h_basis.vir1);
  std::swap(a.h_basis.vir2, b.h_basis.vir2);
  std::swap(a.h_basis.occTau1, b.h_basis.occTau1);
  std::swap(a.h_basis.occTau2, b.h_basis.occTau2);
  std::swap(a.h_basis.virTau1, b.h_basis.virTau1);
  std::swap(a.h_basis.virTau2, b.h_basis.virTau2);
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

  nPrimatives = nprm;
  nShells = nshl;

  if (iops.bopns[KEYS::SPHERICAL]) {
    lspherical = true;
    qc_nbf = nsgs;
  } else {
    lspherical = false;
    qc_nbf = ncgs;
  }

  h_basis.contraction_exp = new double[nPrimatives];
  h_basis.contraction_coef = new double[nPrimatives];
  h_basis.meta_data = new BasisMetaData[nShells];

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
  MPI_Bcast(h_basis.contraction_exp, nPrimatives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.contraction_coef, nPrimatives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(h_basis.meta_data, nShells * sizeof(BasisMetaData), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  normalize();

  /*
  for (i = 0; i < nPrimatives; i++) {
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (mpi_info.taskid == 0) {
        printf("%2i\t%7.3f\t%7.3f\n", i, h_basis.contraction_exp[i], h_basis.contraction_coef[i]);
        fflush(stdout);
      }
  }
  for (i = 0; i < nPrimatives; i++) {
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
  //   std::cout << "NSHL\t" << nShells << "\t" << mpi_info.taskid << std::endl;
  //   std::cout << "NGFS\t" << qc_nbf << "\t" << mpi_info.taskid << std::endl;
  //   std::cout << "NCGS\t" << qc_ncgs << "\t" << mpi_info.taskid << std::endl;
  // }
}

void Basis::normalize() {
  int i, j, k;
  double cnorm, aa, dum, fac, facs, pi32;
  constexpr double pi = 3.141592653589793;

  for (i = 0; i < nShells; i++) {  // number of shells on the atom
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

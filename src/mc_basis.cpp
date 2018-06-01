// Copyright 2017

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "mc_basis.h"
#include "mpi.h"
#include "qc_constant.h"

void MC_Basis::read(MPI_info& mpi_info, Molec& molec, std::string& filename) {
  int znum;
  double alpha[10], coef[10];
  std::string atname;
  std::ifstream input;

  if (mpi_info.sys_master) {
    std::cout << std::endl
              << std::endl;
    std::cout << "Reading MC basis from " << filename << std::endl;
    input.open(filename.c_str());
    if (!input.is_open()) {
      std::cerr << filename << "does not exist" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    input >> mc_nbas >> mc_nprim;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&mc_nbas, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mc_nprim, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (mc_nprim > 10) {
    std::cerr << "Max number of mc basis functions per atom is 10." << std::endl;
    std::cerr << mc_nprim << "basis functions per atom were requested" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  natom = molec.natom;
  mc_basis_list.resize(mc_nbas);
  atom_ibas.resize(natom);

  if (mpi_info.sys_master) {
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\tAtom\t            alpha                         coef" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  }

  for (int i = 0; i < mc_nbas; i++) {
    if (mpi_info.sys_master) {
      input >> atname;                      // read in atom type
      for (int j = 0; j < mc_nprim; j++) {  // read in primativess coefs
        input >> alpha[j] >> coef[j];
      }
      znum = atomic_znum(atname);
      for (int j = 0; j < natom; j++) {    // loop over atoms
        if (znum == molec.atom[j].znum) {  // print out basis functions if atom in molecule
          j = natom;                       // break out of j loop

          // print out basis info
          std::cout << "\t " << atname << "\t";
          std::cout << std::setw(30) << std::setprecision(16) << std::fixed << alpha[0];
          std::cout << std::setw(30) << std::setprecision(16) << std::fixed << coef[0] << std::endl;
          for (int k = 1; k < mc_nprim; k++) {
            std::cout << "\t  \t";
            std::cout << std::setw(30) << std::setprecision(16) << std::fixed << alpha[k];
            std::cout << std::setw(30) << std::setprecision(16) << std::fixed << coef[k] << std::endl;
          }
          std::cout << std::endl;
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&znum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, mc_nprim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coef, mc_nprim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int j = 0; j < mc_nprim; j++) {
      mc_basis_list[i].alpha[j] = alpha[j];
      mc_basis_list[i].norm[j] = exp(0.75 * log(2.0 * alpha[j] / pi)) * coef[j];  // normalize each primative s.t. <phi|phi> is coef[j]^2
    }
    mc_basis_list[i].znum = znum;
  }

  if (mpi_info.sys_master) {
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setprecision(6);
    std::cout << std::endl
              << std::endl;
  }

  for (int i = 0; i < natom; i++) {
    atom_ibas[i] = atom_to_mc_basis(i, molec);
    if (atom_ibas[i] == -1) {
      if (mpi_info.sys_master) {
        std::cerr << "No MC basis function defined for atom with charge " << molec.atom[i].znum << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }

  if (mpi_info.sys_master) {
    input.close();
  }
}

int MC_Basis::atom_to_mc_basis(int param, Molec& molec) {
  for (int i = 0; i < mc_nbas; i++) {
    if (molec.atom[param].znum == mc_basis_list[i].znum) {
      return i;
    }
  }
  return -1;
}

void MC_Basis::mc_eri2v(MPI_info& mpi_info, Molec& molec) {
  int ia, ja, ib, jb, ie, je, m, ts;
  int i, j, k;
  double eri;
  double azi, azj, comz, gzi, normi, normj;
  double rr, tt, cc, h;
  double dummy;
  double dr[3];

  // constants
  double pisub, pisub2;
  double igamma[1501][17];
  double f1[10];
  double f2[] = {-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5};
  double tf[] = {33.0, 37.0, 41.0, 43.0, 46.0, 49.0, 51.0, 54.0, 56.0, 58.0, 61.0, 63.0, 66.0, 68.0, 70.0, 72.0, 74.0};
  pisub = 2.0 / sqrt_pi;
  pisub2 = 2.0 * pow(sqrt_pi, 5);
  f1[0] = 1.0 / pisub;
  f1[1] = 1.0 / 2.0 / pisub;
  f1[2] = 3.0 / 4.0 / pisub;
  f1[3] = 15.0 / 8.0 / pisub;
  f1[4] = 105.0 / 16.0 / pisub;
  f1[5] = 945.0 / 32.0 / pisub;
  f1[6] = 10395.0 / 64.0 / pisub;
  f1[7] = 135135.0 / 128.0 / pisub;
  f1[8] = 2027025.0 / 256.0 / pisub;
  f1[9] = 34459425.0 / 512.0 / pisub;
  for (i = 0; i <= 1500; i++) {
    for (j = 0; j < 17; j++) {
      igamma[i][j] = 0.00;
    }
  }
  for (i = 0; i <= 1500; i++) {
    double a = static_cast<double>(i) / 20.0;
    double b = exp(-a);
    double c = 0.0;
    // k = 50 + int(i/3.0); org
    k = 50 + i / 3.0;  // might be broke
    for (j = k; j >= 0; j--) {
      c = (2.0 * a * c + b) / static_cast<double>(2 * j + 1);
      if (j <= 16) {
        igamma[i][j] = c;
      }
    }
  }

  // calculate g_wgt
  g_wgt = 0.0;
  for (ia = 0; ia < natom; ia++) {  // --- electron (1) ----
    ib = atom_ibas[ia];

    for (ja = 0; ja < natom; ja++) {  // --- electron (2) ----
      jb = atom_ibas[ja];

      dr[0] = molec.atom[ia].pos[0] - molec.atom[ja].pos[0];
      dr[1] = molec.atom[ia].pos[1] - molec.atom[ja].pos[1];
      dr[2] = molec.atom[ia].pos[2] - molec.atom[ja].pos[2];
      rr = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

      for (ie = 0; ie < mc_nprim; ie++) {
        azi = mc_basis_list[ib].alpha[ie];
        normi = mc_basis_list[ib].norm[ie];

        for (je = 0; je < mc_nprim; je++) {
          azj = mc_basis_list[jb].alpha[je];
          normj = mc_basis_list[jb].norm[je];

          comz = azi + azj;
          gzi = azi * azj / comz;

          tt = gzi * rr;
          cc = pisub2 / (sqrt(comz) * azi * azj);

          // for only S
          m = 0;
          if (tt < tf[m]) {
            ts = round(tt * 20.0);
            h = 0.05 * static_cast<double>(ts) - tt;
            dummy = igamma[ts][m + 6] * h * 0.166666666666667 + igamma[ts][m + 5];
            dummy = dummy * h * 0.20 + igamma[ts][m + 4];
            dummy = dummy * h * 0.25 + igamma[ts][m + 3];
            dummy = dummy * h / 3.00 + igamma[ts][m + 2];
            dummy = dummy * h * 0.50 + igamma[ts][m + 1];
            dummy = dummy * h + igamma[ts][m + 0];
            eri = cc * dummy;
          } else {
            eri = cc * f1[m] * exp(f2[m] * log(tt));
          }
          eri = eri * normi * normj;
          g_wgt = g_wgt + eri;
        }
      }
    }
  }

  if (mpi_info.sys_master) {
    std::cout << "g_wgt\t" << g_wgt << std::endl;
  }

  for (ib = 0; ib < mc_nbas; ib++) {
    for (ie = 0; ie < mc_nprim; ie++) {
      mc_basis_list[ib].norm[ie] = mc_basis_list[ib].norm[ie] / sqrt(g_wgt);
    }
  }
}

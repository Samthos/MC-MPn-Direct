// Copyright 2017

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef HAVE_MPI
#include "mpi.h"
#endif
#include "qc_basis.h"
#include "../atom_znum.h"


SHELL::Shell_Type SHELL::string_to_shell_type(const std::string& str) {
  if (str == "SP") {
    return SHELL::SP;
  } else if (str == "S") {
    return SHELL::S;
  } else if (str == "P") {
    return SHELL::P;
  } else if (str == "D") {
    return SHELL::D;
  } else if (str == "F") {
    return SHELL::F;
  } else if (str == "G") {
    return SHELL::G;
  } else if (str == "H") {
    return SHELL::H;
  } else {
    std::cerr << "When trying to read basis encountered usuported angular momentum of " << str << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

Basis::Basis(IOPs &iops, MPI_info &mpi_info, Molec &molec) {
  /*
   * Basis constructor
   *
   * Arguments:
   *  IOPS iops: see qc_input.h
   *  MPI_info mpi_info: see qc_mpi.h
   *  Molec molec: geometry of molecule
   */

  // read basis set
#ifdef OLD_BASIS_READ
  read(iops, mpi_info, molec);
#else
  read_new(iops, mpi_info, molec);
#endif
  dump(iops.sopns[KEYS::JOBNAME]);

  // molecular orbital coefficients and energies
  nw_vectors_read(iops, mpi_info, molec);

  // declare memory
  mc_pair_num = iops.iopns[KEYS::MC_NPAIR];
  h_basis.ao_amplitudes = new double[nw_nbf * mc_pair_num];
  h_basis.psi1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psi2 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau1 = new double[(ivir2-iocc1) * mc_pair_num];
  h_basis.psiTau2 = new double[(ivir2-iocc1) * mc_pair_num];

  // set convince pointers
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
  /*
   * Basis destructor
   */
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
  /*
   * Copy constructor
   */
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
  /*
   * copy assignment operator
   */
  std::swap(*this, param);
  return *this;
}
void swap(Basis& a, Basis& b) {
  /*
   * swap operator
   */
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
  /*
   * reads a basis set; see TKpathTK for an example
   *
   * Arguments:
   *  IOPS iops: see qc_input.h
   *  MPI_info mpi_info: see qc_mpi.h
   *  Molec molec: geometry of molecule
   */
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
              h_basis.meta_data[shell].angular_momentum = 0;
              ao_offset = 1;
            } else if (sym == "P") {
              h_basis.meta_data[shell].angular_momentum = 1;
              ao_offset = 3;
            } else if (sym == "D") {
              h_basis.meta_data[shell].angular_momentum = 2;
              ao_offset = 5;
            } else if (sym == "F") {
              h_basis.meta_data[shell].angular_momentum = 3;
              ao_offset = 7;
            } else if (sym == "G") {
              h_basis.meta_data[shell].angular_momentum = 4;
              ao_offset = 9;
            }

            if (h_basis.meta_data[shell].angular_momentum == -1) {
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

void Basis::read_new(IOPs& iops, MPI_info& mpi_info, Molec& molec) {
  std::ifstream input;
  std::string str;
  std::string atomName, basisName, basisType, shell_type;

  int currentBasis = -1;
  std::vector<AtomBasis> atomBasis(100);

  lspherical = iops.bopns[KEYS::SPHERICAL];

  if (mpi_info.sys_master) {
    // input.open(iops.sopns[KEYS::BASIS].c_str());
    std::cout << "Basis set: " << iops.sopns[KEYS::BASIS] << std::endl;
    input.open(iops.sopns[KEYS::BASIS].c_str());
    if (input.good() && input.is_open()) {
      while (getline(input, str)) {
        std::stringstream ss;
        ss << str;
        if (str.compare(0, 1, "#") == 0)  // commented line
        {
        } else if (str.compare(0, 5, "basis") ==
            0)  // start of basis data for an atom
        {
          ss.ignore(5);
          ss >> basisName >> basisType;

          // remove initial and final quote from basisName
          basisName = basisName.substr(1, basisName.length() - 2);

          // extract atom type from basis Name
          std::size_t pos = basisName.find('_');
          atomName = basisName.substr(0, pos);

          // find charge of atom
          currentBasis = atomic_znum(atomName);

          // store information in atomBasis;
          atomBasis[currentBasis].atomName = atomName;
          atomBasis[currentBasis].basisName = basisName;
          atomBasis[currentBasis].basisType = basisType;
          atomBasis[currentBasis].atomCharge = currentBasis;
        } else if (str.compare(0, 3, "end") == 0) { // end of basis data for an atom
          currentBasis = -1;
        } else if (currentBasis > 0) {
          if (str.at(0) >= 65 && str.at(0) <= 90)  { // start of a new shell
            ss >> atomName >> shell_type;

            SHELL::Shell tempShell;
            tempShell.shell_type = SHELL::string_to_shell_type(shell_type);
            atomBasis[currentBasis].shell.push_back(tempShell);
          } else { // reading a shell
            double alpha, tempCoef;
            std::vector<double> coef;

            ss >> alpha;
            while (ss >> tempCoef) {
              coef.push_back(tempCoef);
            }
            atomBasis[currentBasis].shell.back().contracted_gaussian.emplace_back(alpha, coef);
          }
        }
      }
    } else {
      std::cerr << "Basis set " << iops.sopns[KEYS::BASIS] << " not found" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    normalize_sperical_atom_basis(atomBasis);

    // count number of primatives is basis set
    qc_nbf = 0;
    nShells = 0;
    nPrimatives = 0;
    for (auto& it : molec.atom) {
      for (auto& jt : atomBasis[it.znum].shell) {
        nShells += jt.contracted_gaussian.front().second.size();
        nPrimatives += jt.contracted_gaussian.size() * jt.contracted_gaussian.front().second.size();
        if (lspherical) {
          if (jt.shell_type == SHELL::S) {
            qc_nbf += jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::P) {
            qc_nbf += 3 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::D) {
            qc_nbf += 5 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::F) {
            qc_nbf += 7 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::G) {
            qc_nbf += 9 * jt.contracted_gaussian.front().second.size();
          }
        } else {
          if (jt.shell_type == SHELL::S) {
            qc_nbf += jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::P) {
            qc_nbf += 3 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::D) {
            qc_nbf += 6 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::F) {
            qc_nbf += 10 * jt.contracted_gaussian.front().second.size();
          } else if (jt.shell_type == SHELL::G) {
            qc_nbf += 15 * jt.contracted_gaussian.front().second.size();
          }
        }
      }
    }
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&qc_nshl, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&qc_nprm, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&qc_nbf, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  h_basis.contraction_exp = new double[nPrimatives];
  h_basis.contraction_coef = new double[nPrimatives];
  h_basis.meta_data = new BasisMetaData[nShells];

  if (mpi_info.sys_master) {
    uint contraction= 0;
    for (auto& atom : molec.atom) {
      for (auto& shell : atomBasis[atom.znum].shell) {
        for (uint shell_contraction = 0; shell_contraction < shell.n_contractions(); shell_contraction++) {
          // set start and stop
          if (contraction == 0) {
            h_basis.meta_data[contraction].contraction_begin = 0;
          } else {
            h_basis.meta_data[contraction].contraction_begin = h_basis.meta_data[contraction - 1].contraction_end;
          }
          h_basis.meta_data[contraction].contraction_end = h_basis.meta_data[contraction].contraction_begin + shell.contracted_gaussian.size();

          // set atom
          std::copy(atom.pos, atom.pos + 3, h_basis.meta_data[contraction].pos);

          // set angular momentum
          {
            if (shell.shell_type == SHELL::SP) {
              h_basis.meta_data[contraction].angular_momentum = -1;
            } else if (shell.shell_type == SHELL::S) {
              h_basis.meta_data[contraction].angular_momentum = 0;
            } else if (shell.shell_type == SHELL::P) {
              h_basis.meta_data[contraction].angular_momentum = 1;
            } else if (shell.shell_type == SHELL::D) {
              h_basis.meta_data[contraction].angular_momentum = 2;
            } else if (shell.shell_type == SHELL::F) {
              h_basis.meta_data[contraction].angular_momentum = 3;
            } else if (shell.shell_type == SHELL::G) {
              h_basis.meta_data[contraction].angular_momentum = 4;
            }
          }

          // set isgs
          if (contraction == 0) {
            h_basis.meta_data[contraction].ao_begin = 0;
          } else {
            if (lspherical) {
              if (h_basis.meta_data[contraction - 1].angular_momentum >= 0) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 2 * h_basis.meta_data[contraction - 1].angular_momentum + 1;
              } else {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 4;
              }
            } else {
              if (h_basis.meta_data[contraction - 1].angular_momentum == -1) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 4;
              } else if (h_basis.meta_data[contraction - 1].angular_momentum == 0) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 1;
              } else if (h_basis.meta_data[contraction - 1].angular_momentum == 1) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 3;
              } else if (h_basis.meta_data[contraction - 1].angular_momentum == 2) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 6;
              } else if (h_basis.meta_data[contraction - 1].angular_momentum == 3) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 10;
              } else if (h_basis.meta_data[contraction - 1].angular_momentum == 4) {
                h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin + 15;
              }
            }
          }

          // copy alpha and norm
          for (auto kt = shell.contracted_gaussian.begin(); kt != shell.contracted_gaussian.end(); kt++) {
            auto k = std::distance(shell.contracted_gaussian.begin(), kt) + h_basis.meta_data[contraction].contraction_begin;
            h_basis.contraction_exp[k] = kt->first;
            h_basis.contraction_coef[k] = kt->second[shell_contraction];
          }
          contraction++;
        }
      }
    }
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(alpha.data(), alpha.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(norm.data(), norm.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(basisMetaData.data(), basisMetaData.size() * sizeof(BasisMetaData),
            MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
}

void Basis::normalize() {
  int i, j, k;
  double cnorm, aa, dum, fac, facs, pi32;
  constexpr double pi = 3.141592653589793;

  for (i = 0; i < nShells; i++) {  // number of shells on the atom
    if (h_basis.meta_data[i].angular_momentum == -1) {
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
    } else if (h_basis.meta_data[i].angular_momentum == 0) {
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
    } else if (h_basis.meta_data[i].angular_momentum == 1) {
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
    } else if (h_basis.meta_data[i].angular_momentum == 2) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi)) * 4.0 * h_basis.contraction_exp[j];
        cnorm = cnorm / sqrt(3.0);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;  // dxx
      }
      // ncgs = ncgs + 6;
      // nsgs[0] = nsgs[0] + 5;
    } else if (h_basis.meta_data[i].angular_momentum == 3) {
      for (j = h_basis.meta_data[i].contraction_begin; j < h_basis.meta_data[i].contraction_end; j++) {
        cnorm = exp(0.75 * log(2.0 * h_basis.contraction_exp[j] / pi)) * pow(4.0 * h_basis.contraction_exp[j], 1.5);
        cnorm = cnorm / sqrt(15.0);
        h_basis.contraction_coef[j] = h_basis.contraction_coef[j] * cnorm;  // dxx
      }

      // ncgs = ncgs + 10;
      // nsgs[0] = nsgs[0] + 7;
    } else if (h_basis.meta_data[i].angular_momentum == 4) {
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

void Basis::normalize_sperical_atom_basis(std::vector<AtomBasis>& atomBasis) {
  constexpr double pi = 3.141592653589793;
  double cnorm, aa, dum, fac, facs, pi32;
  std::vector<double> norm;

  // loop of atomic basis sets
  for (auto& it : atomBasis) {
    // if basis set was read
    if (it.atomCharge != -1) {
      // loop over shells
      for (auto& shell : it.shell) {
        if (shell.shell_type == SHELL::SP)  { // SP shell
          /*
          //qc_shl_list[nshl[0]].ncgs = 4;
          //qc_shl_list[nshl[0]].nsgs = 4;

          //qc_shl_list[nshl[0]].h_basis.norm = new double*[2];
          //for(j=0;j<2;j++) {
          //	qc_shl_list[nshl[0]].h_basis.norm[j] = new double[nprim[i]];
          //}

          //nsgs = nsgs + 4;
          //ncgs = ncgs + 4;

          //loop over primatives
          for( unsigned int i = 0; i < shell.alpha.size(); i++)
          {
                  //loop over sets
                  for(auto& kt : shell.coef[i])
                  {
                  }
                  for(j=0;j<nprim[i];j++) {
                          cnorm = exp(0.75 * log(2.0 * h_basis.alpha[j][i] /
          pi));
                          qc_shl_list[nshl[0]].h_basis.alpha[j]  =
          h_basis.alpha[j][i];
                          qc_shl_list[nshl[0]].h_basis.norm[1][j] = coef[j][i] *
          ch_basis.norm;
                          cnorm = cnorm * sqrt(4.0 * new_alpha[j][i]);
                          qc_shl_list[nshl[0]].new_norm[2][j]  = coef2[j][i] *
          cnorm;
                  }
          }
          */
        } else if (shell.shell_type == SHELL::S) {
          // normalize each gaussion in contraction
          for (auto & kt : shell.contracted_gaussian) {
            cnorm = pow(2.0 * kt.first / pi, 0.75);
            for (double & lt : kt.second) {
              lt *= cnorm;
            }
          }

          // calculate normalization of each contraction
          norm.resize(shell.n_contractions());
          std::fill(norm.begin(), norm.end(), 0);
          for (auto first_guassian = shell.contracted_gaussian.begin(); first_guassian != shell.contracted_gaussian.end(); first_guassian++) {
            for (auto second_guassian = first_guassian; second_guassian != shell.contracted_gaussian.end(); second_guassian++) {
              for (auto m = 0; m < norm.size(); m++) {
                aa = first_guassian->first + second_guassian->first;
                fac = aa * sqrt(aa);
                dum = first_guassian->second[m] * second_guassian->second[m] / fac;
                if (first_guassian != second_guassian) {
                  dum = dum + dum;
                }
                norm[m] += dum;
              }
            }
          }
          pi32 = 5.56832799683170;
          for (auto& kt : norm) {
            kt = 1.0 / sqrt(kt * pi32);
          }

          // apply normalization to each contraction
          for (auto & kt : shell.contracted_gaussian) {
            for (auto l = 0; l < kt.second.size(); l++) {
              kt.second[l] *= norm[l];
            }
          }
        } else if (shell.shell_type == SHELL::P) {
          // normalize each guassian
          for (auto & kt : shell.contracted_gaussian) {
            cnorm = sqrt(4.0 * kt.first) * pow(2.0 * kt.first / pi, 0.75);
            for (double & lt : kt.second) {
              lt *= cnorm;
            }
          }

          norm.resize(shell.n_contractions());
          std::fill(norm.begin(), norm.end(), 0);
          for (auto first_guassian = shell.contracted_gaussian.begin(); first_guassian != shell.contracted_gaussian.end(); first_guassian++) {
            for (auto second_guassian = first_guassian; second_guassian != shell.contracted_gaussian.end(); second_guassian++) {
              for (auto m = 0; m < norm.size(); m++) {
                aa = first_guassian->first + second_guassian->first;
                fac = aa * aa * sqrt(aa);
                dum = 0.5 * first_guassian->second[m] * second_guassian->second[m] / fac;
                if (first_guassian != second_guassian) {
                  dum = dum + dum;
                }
                norm[m] += dum;
              }
            }
          }
          pi32 = 5.56832799683170;
          for (auto& kt : norm) {
            kt = 1.0 / sqrt(kt * pi32);
          }

          for (auto & kt : shell.contracted_gaussian) {
            for (auto l = 0; l < kt.second.size(); l++) {
              kt.second[l] *= norm[l];
            }
          }
        } else if (shell.shell_type == SHELL::D) {
          for (auto & kt : shell.contracted_gaussian) {
            cnorm = pow(2.0 * kt.first / pi, 0.75) * 4.0 * kt.first / sqrt(3.0);
            for (double & lt : kt.second) {
              lt *= cnorm;
            }
          }
        } else if (shell.shell_type == SHELL::F) {
          for (auto & kt : shell.contracted_gaussian) {
            cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first, 1.5) / sqrt(15.0);
            for (double & lt : kt.second) {
              lt *= cnorm;
            }
          }
        } else if (shell.shell_type == SHELL::G) {
          for (auto & kt : shell.contracted_gaussian) {
            cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first,2.0) / sqrt(105.0);
            for (double &lt : kt.second) {
              lt *= cnorm;
            }
          }
        }
      }
    }
  }
}

void Basis::dump(const std::string& fname) {
  std::ofstream os(fname);
  os << "\n-----------------------------------------------------------------------------------------------------------\nBasis Dump\n";
  os << "mc_pair_num: " << mc_pair_num << "\n";
  os << "iocc1: " << iocc1 << "\n";        // index first occupied orbital to be used
  os << "iocc2: " << iocc2 << "\n";        // index of HOMO+1
  os << "ivir1: " << ivir1 << "\n";        // index of LUMO
  os << "ivir2: " << ivir2 << "\n";        // index of last virtual to be used + 1
  os << "qc_nbf: " << qc_nbf << "\n";       // number basis functions
  os << "nShells: " << nShells << "\n";      // number of shells
  os << "nPrimatives: " << nPrimatives << "\n";  // number of primitives
  os << "lspherical: " << lspherical << "\n";  // true if spherical
  for (int i = 0; i < nPrimatives; ++i) {
    os << h_basis.contraction_coef[i] << "\t" << h_basis.contraction_exp[i] << "\n";
  }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

// Copyright 2017

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../qc_mpi.h"
#include "qc_basis.h"
#include "../atom_tag_parser.h"


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
int SHELL::number_of_polynomials(SHELL::Shell_Type shell_type, bool spherical) {
  if (spherical) {
    return SHELL::number_of_spherical_polynomials(shell_type);
  } else {
    return SHELL::number_of_cartesian_polynomials(shell_type);
  }
}
int SHELL::number_of_polynomials(int shell_type, bool spherical) {
  return number_of_polynomials(static_cast<SHELL::Shell_Type>(shell_type), spherical);
}
int SHELL::number_of_spherical_polynomials(SHELL::Shell_Type shell_type) {
  int nbf = 0;
  if (shell_type == SHELL::SP) {
    nbf += 4;
  } else {
    nbf += 2 * shell_type + 1;
  }
  return nbf;
}
int SHELL::number_of_cartesian_polynomials(SHELL::Shell_Type shell_type) {
  int nbf = 0;
  if (shell_type == SHELL::SP) {
    nbf = 4;
  } else {
    int l = shell_type;
    nbf = (l+1) * (l+2) / 2;
  }
  return nbf;
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
  read(iops, mpi_info, molec);

  // molecular orbital coefficients and energies
  nw_vectors_read(iops, mpi_info, molec);

  // declare memory
  mc_pair_num = iops.iopns[KEYS::MC_NPAIR];
  h_basis.ao_amplitudes = new double[nw_nbf * mc_pair_num];
  h_basis.contraction_amplitudes = new double[nShells * mc_pair_num];
  h_basis.contraction_amplitudes_derivative = new double[nShells * mc_pair_num];
}
Basis::~Basis() {
  /*
   * Basis destructor
   */
  delete[] nw_en;

  delete[] h_basis.nw_co;
  delete[] h_basis.ao_amplitudes;
  delete[] h_basis.contraction_amplitudes;
  delete[] h_basis.contraction_amplitudes_derivative;
  delete[] h_basis.contraction_exp;
  delete[] h_basis.contraction_coef;
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

  nw_en = new double[nw_nbf];
  std::copy(param.nw_en, param.nw_en + nw_nbf, nw_en);

  h_basis.nw_co = new double[nw_nbf * nw_nmo];
  std::copy(param.h_basis.nw_co, param.h_basis.nw_co + nw_nmo * nw_nbf, h_basis.nw_co);

  h_basis.contraction_amplitudes = new double[nShells * mc_pair_num];
  h_basis.contraction_amplitudes_derivative = new double[nShells * mc_pair_num];
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
  std::swap(a.h_basis.contraction_amplitudes, b.h_basis.contraction_amplitudes);
  std::swap(a.h_basis.contraction_amplitudes_derivative, b.h_basis.contraction_amplitudes_derivative);
  std::swap(a.h_basis.contraction_exp, b.h_basis.contraction_exp);
  std::swap(a.h_basis.contraction_coef, b.h_basis.contraction_coef);
}

void Basis::read(IOPs& iops, MPI_info& mpi_info, Molec& molec) {
  std::ifstream input;
  std::string str;
  std::string atomName, basisName, basisType, shell_type;

  Atom_Tag_Parser atom_tag_parser;

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
        if (str.compare(0, 1, "#") == 0) { // commented line
        } else if (str.compare(0, 5, "basis") == 0) { // start of basis data for an atom
          ss.ignore(5);
          ss >> basisName >> basisType;

          // remove initial and final quote from basisName
          basisName = basisName.substr(1, basisName.length() - 2);

          // extract atom type from basis Name
          std::size_t pos = basisName.find('_');
          atomName = basisName.substr(0, pos);

          // find charge of atom
          currentBasis = atom_tag_parser.parse(atomName);

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

    normalize_atom_basis(atomBasis);

    // count number of primatives is basis set
    qc_nbf = 0;
    nShells = 0;
    nPrimatives = 0;
    for (auto& it : molec.atoms) {
      for (auto& jt : atomBasis[it.znum].shell) {
        nShells += jt.contracted_gaussian.front().second.size();
        nPrimatives += jt.contracted_gaussian.size() * jt.contracted_gaussian.front().second.size();
        if (jt.shell_type == SHELL::SP) {
          qc_nbf += number_of_polynomials(jt.shell_type, lspherical);
        } else {
          qc_nbf += number_of_polynomials(jt.shell_type, lspherical) * jt.contracted_gaussian.front().second.size();
        }
      }
    }
  }

  MPI_info::barrier();
  MPI_info::broadcast_int(&nShells, 1);
  MPI_info::broadcast_int(&nPrimatives, 1);
  MPI_info::broadcast_int(&qc_nbf, 1);

  h_basis.contraction_exp = new double[nPrimatives];
  h_basis.contraction_coef = new double[nPrimatives];
  h_basis.meta_data = new BasisMetaData[nShells];

  if (mpi_info.sys_master) {
    uint contraction= 0;
    for (auto& atom : molec.atoms) {
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
          if (shell.shell_type == SHELL::SP) {
            if (shell_contraction == 0) {
              h_basis.meta_data[contraction].angular_momentum = 0;
            } else {
              h_basis.meta_data[contraction].angular_momentum = 1;
            }
          } else {
            h_basis.meta_data[contraction].angular_momentum = shell.shell_type;
          }

          // set isgs
          if (contraction == 0) {
            h_basis.meta_data[contraction].ao_begin = 0;
          } else {
            h_basis.meta_data[contraction].ao_begin = h_basis.meta_data[contraction - 1].ao_begin
                + SHELL::number_of_polynomials(h_basis.meta_data[contraction - 1].angular_momentum, lspherical);
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

  MPI_info::barrier();
  MPI_info::broadcast_double(h_basis.contraction_exp, nPrimatives);
  MPI_info::broadcast_double(h_basis.contraction_coef, nPrimatives);
  MPI_info::broadcast_char((char*) h_basis.meta_data, nShells * sizeof(BasisMetaData));
}

void Basis::normalize_atom_basis(std::vector<AtomBasis>& atomBasis) {
  // loop of atomic basis sets
  for (auto& it : atomBasis) {
    // if basis set was read
    if (it.atomCharge != -1) {
      // loop over shells
      for (auto& shell : it.shell) {
        switch (shell.shell_type) {
          case SHELL::SP: normalize_sp(shell); break;
          case SHELL::S: normalize_s(shell); break;
          case SHELL::P: normalize_p(shell); break;
          case SHELL::D: normalize_d(shell); break;
          case SHELL::F: normalize_f(shell); break;
          case SHELL::G: normalize_g(shell); break;
        }
      }
    }
  }
}
void Basis::normalize_sp(SHELL::Shell& shell) {
  for (int j = 0; j < 2; ++j) {
    SHELL::Shell temp_shell;
    temp_shell.contracted_gaussian.resize(shell.contracted_gaussian.size());
    for (int i = 0; i < shell.contracted_gaussian.size(); i++){
      temp_shell.contracted_gaussian[i].first = shell.contracted_gaussian[i].first;
      temp_shell.contracted_gaussian[i].second.resize(1);
      temp_shell.contracted_gaussian[i].second[0] = shell.contracted_gaussian[i].second[j];
    }

    if (j == 0) {
      normalize_s(temp_shell);
    } else {
      normalize_p(temp_shell);
    }

    for (int i = 0; i < shell.contracted_gaussian.size(); i++){
      shell.contracted_gaussian[i].second[j] = temp_shell.contracted_gaussian[i].second[0];
    }
  }
}
void Basis::normalize_s(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;
  constexpr double pi32 = 5.56832799683170;

  // normalize each gaussion in contraction
  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }

  // calculate normalization of each contraction
  std::vector<double> norm(shell.n_contractions(), 0.00);
  for (auto first_guassian = shell.contracted_gaussian.begin(); first_guassian != shell.contracted_gaussian.end(); first_guassian++) {
    for (auto second_guassian = first_guassian; second_guassian != shell.contracted_gaussian.end(); second_guassian++) {
      for (auto m = 0; m < norm.size(); m++) {
        double fac = pow(first_guassian->first + second_guassian->first, 1.5);
        double dum = first_guassian->second[m] * second_guassian->second[m] / fac;
        if (first_guassian != second_guassian) {
          dum = dum + dum;
        }
        norm[m] += dum;
      }
    }
  }
  for (auto& kt : norm) {
    kt = 1.0 / sqrt(kt * pi32);
  }

  // apply normalization to each contraction
  for (auto & kt : shell.contracted_gaussian) {
    for (auto l = 0; l < kt.second.size(); l++) {
      kt.second[l] *= norm[l];
    }
  }
}
void Basis::normalize_p(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;
  constexpr double pi32 = 5.56832799683170;

  // normalize each guassian
  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = sqrt(4.0 * kt.first) * pow(2.0 * kt.first / pi, 0.75);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }

  std::vector<double> norm(shell.n_contractions(), 0.00);
  for (auto first_guassian = shell.contracted_gaussian.begin(); first_guassian != shell.contracted_gaussian.end(); first_guassian++) {
    for (auto second_guassian = first_guassian; second_guassian != shell.contracted_gaussian.end(); second_guassian++) {
      for (auto m = 0; m < norm.size(); m++) {
        double fac = pow(first_guassian->first + second_guassian->first, 2.5);
        double dum = 0.5 * first_guassian->second[m] * second_guassian->second[m] / fac;
        if (first_guassian != second_guassian) {
          dum = dum + dum;
        }
        norm[m] += dum;
      }
    }
  }
  for (auto& kt : norm) {
    kt = 1.0 / sqrt(kt * pi32);
  }

  for (auto & kt : shell.contracted_gaussian) {
    for (auto l = 0; l < kt.second.size(); l++) {
      kt.second[l] *= norm[l];
    }
  }
}
void Basis::normalize_d(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * 4.0 * kt.first / sqrt(3.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void Basis::normalize_f(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first, 1.5) / sqrt(15.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void Basis::normalize_g(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first,2.0) / sqrt(105.0);
    for (double &lt : kt.second) {
      lt *= cnorm;
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
  os << "qc_nbf: " << qc_nbf << "\t" << nw_nbf << "\n";       // number basis functions
  os << "nShells: " << nShells << "\n";      // number of shells
  os << "nPrimatives: " << nPrimatives << "\n";  // number of primitives
  os << "lspherical: " << lspherical << "\n";  // true if spherical
  for (int i = 0; i < nPrimatives; ++i) {
    os << h_basis.contraction_coef[i] << "\t" << h_basis.contraction_exp[i] << "\n";
  }
  for (int i = 0; i < nShells; ++i) {
    os << h_basis.meta_data[i].ao_begin << "\t";
    os << h_basis.meta_data[i].contraction_begin << "\t";
    os << h_basis.meta_data[i].contraction_end << "\t";
    os << h_basis.meta_data[i].angular_momentum << "\t";
    os << h_basis.meta_data[i].pos[0] << "\t";
    os << h_basis.meta_data[i].pos[1] << "\t";
    os << h_basis.meta_data[i].pos[2] << "\n";
  }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

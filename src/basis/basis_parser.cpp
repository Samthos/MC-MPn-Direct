#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

#include "../atom_tag_parser.h"

#include "basis_parser.h"

Basis_Parser::Basis_Parser(IOPs &iops, MPI_info &mpi_info, Molec &molec) {
  read(iops, mpi_info, molec);
}

void Basis_Parser::read(IOPs& iops, MPI_info& mpi_info, Molec& molec) {
  std::ifstream input;
  std::string str;
  std::string atomName, basisName, basisType, shell_type;

  Atom_Tag_Parser atom_tag_parser;

  int currentBasis = -1;
  std::vector<AtomBasis> atomBasis(100);

  is_spherical = iops.bopns[KEYS::SPHERICAL];

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
    n_atomic_orbitals = 0;
    n_shells = 0;
    n_primatives = 0;
    for (auto& it : molec.atoms) {
      for (auto& jt : atomBasis[it.znum].shell) {
        n_shells += jt.contracted_gaussian.front().second.size();
        n_primatives += jt.contracted_gaussian.size() * jt.contracted_gaussian.front().second.size();
        if (jt.shell_type == SHELL::SP) {
          n_atomic_orbitals += number_of_polynomials(jt.shell_type, is_spherical);
        } else {
          n_atomic_orbitals += number_of_polynomials(jt.shell_type, is_spherical) * jt.contracted_gaussian.front().second.size();
        }
      }
    }
  }

  MPI_info::barrier();
  MPI_info::broadcast_int(&n_shells, 1);
  MPI_info::broadcast_int(&n_primatives, 1);
  MPI_info::broadcast_int(&n_atomic_orbitals, 1);

  contraction_coeficients.resize(n_primatives);
  contraction_exponents.resize(n_primatives);
  atomic_orbitals.resize(n_shells);

  if (mpi_info.sys_master) {
    uint contraction= 0;
    for (auto& atom : molec.atoms) {
      for (auto& shell : atomBasis[atom.znum].shell) {
        for (uint shell_contraction = 0; shell_contraction < shell.n_contractions(); shell_contraction++) {
          // set start and stop
          if (contraction == 0) {
            atomic_orbitals[contraction].contraction_begin = 0;
          } else {
            atomic_orbitals[contraction].contraction_begin = atomic_orbitals[contraction - 1].contraction_end;
          }
          atomic_orbitals[contraction].contraction_end = atomic_orbitals[contraction].contraction_begin + shell.contracted_gaussian.size();

          // set atom
          std::copy(atom.pos, atom.pos + 3, atomic_orbitals[contraction].pos);

          // set angular momentum
          if (shell.shell_type == SHELL::SP) {
            if (shell_contraction == 0) {
              atomic_orbitals[contraction].angular_momentum = 0;
            } else {
              atomic_orbitals[contraction].angular_momentum = 1;
            }
          } else {
            atomic_orbitals[contraction].angular_momentum = shell.shell_type;
          }

          // set isgs
          if (contraction == 0) {
            atomic_orbitals[contraction].ao_begin = 0;
          } else {
            atomic_orbitals[contraction].ao_begin = atomic_orbitals[contraction - 1].ao_begin
                + SHELL::number_of_polynomials(atomic_orbitals[contraction - 1].angular_momentum, is_spherical);
          }

          // copy alpha and norm
          for (auto kt = shell.contracted_gaussian.begin(); kt != shell.contracted_gaussian.end(); kt++) {
            auto k = std::distance(shell.contracted_gaussian.begin(), kt) + atomic_orbitals[contraction].contraction_begin;
            contraction_exponents[k] = kt->first;
            contraction_coeficients[k] = kt->second[shell_contraction];
          }
          contraction++;
        }
      }
    }
  }

  MPI_info::barrier();
  MPI_info::broadcast_vector_double(contraction_exponents);
  MPI_info::broadcast_vector_double(contraction_coeficients);
  MPI_info::broadcast_char((char*) atomic_orbitals.data(), n_shells * sizeof(BasisMetaData));
}
void Basis_Parser::normalize_atom_basis(std::vector<AtomBasis>& atomBasis) {
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
void Basis_Parser::normalize_sp(SHELL::Shell& shell) {
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
void Basis_Parser::normalize_s(SHELL::Shell& shell) {
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
void Basis_Parser::normalize_p(SHELL::Shell& shell) {
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
void Basis_Parser::normalize_d(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * 4.0 * kt.first / sqrt(3.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void Basis_Parser::normalize_f(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first, 1.5) / sqrt(15.0);
    for (double & lt : kt.second) {
      lt *= cnorm;
    }
  }
}
void Basis_Parser::normalize_g(SHELL::Shell& shell) {
  constexpr double pi = 3.141592653589793;

  for (auto & kt : shell.contracted_gaussian) {
    double cnorm = pow(2.0 * kt.first / pi, 0.75) * pow(4.0 * kt.first,2.0) / sqrt(105.0);
    for (double &lt : kt.second) {
      lt *= cnorm;
    }
  }
}

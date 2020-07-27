#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

#include "atom_tag_parser.h"

#include "basis_parser.h"

Basis_Parser::Basis_Parser() :  atomBasis(100) { }

Basis_Parser::Basis_Parser(const std::string& basis_name_in, bool is_spherical_in,  MPI_info &mpi_info, Molecule &molec) :
  is_spherical(is_spherical_in),
  basis_name(basis_name_in),
  atomBasis(100)
{
  read(mpi_info);
  build_atomic_orbitals(mpi_info, molec);
}

void Basis_Parser::read(MPI_info& mpi_info) {
  std::ifstream input;
  std::string str;
  std::string atomName;
  std::string basisName;
  std::string basisType;
  std::string shell_type_string;

  Atom_Tag_Parser atom_tag_parser;

  int currentBasis = -1;

  SHELL::Shell_Type shell_type = SHELL::NA;
  SHELL::Contracted_Gaussian contracted_gaussian;

  if (mpi_info.sys_master) {
    std::cout << "Basis set: " << basis_name << std::endl;
    input.open(basis_name);
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
          atomBasis[currentBasis].shell.emplace_back(shell_type, contracted_gaussian);
          shell_type = SHELL::NA;
          currentBasis = -1;
        } else if (currentBasis > 0) {
          if (str.at(0) >= 65 && str.at(0) <= 90)  { // start of a new shell
            if (shell_type != SHELL::NA) {
              try {
                atomBasis[currentBasis].shell.emplace_back(shell_type, contracted_gaussian);
              } catch (SHELL::Shell_Exception &e) {
                std::cerr << e.what() << "\n";
                std::cerr << currentBasis << "\n";
                std::cerr << atomName << "\n";
                std::cerr.flush();
                throw e;
              }
            }
            ss >> atomName >> shell_type_string;

            shell_type = SHELL::string_to_shell_type(shell_type_string);
            contracted_gaussian.clear();
          } else { // reading a shell
            double alpha, tempCoef;
            std::vector<double> coef;

            ss >> alpha;
            while (ss >> tempCoef) {
              coef.push_back(tempCoef);
            }
            contracted_gaussian.emplace_back(alpha, coef);
          }
        }
      }
    } else {
      std::cerr << "Basis set " << basis_name << " not found" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

void Basis_Parser::build_atomic_orbitals(MPI_info& mpi_info, Molecule& molec) {
  if (mpi_info.sys_master) {
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

  if (mpi_info.sys_master) {
    uint contraction= 0;
    SHELL::Shell_Type shell_type = SHELL::NA;
    int contraction_begin = 0;
    int contraction_end = 0;
    int contraction_index = 0;
    int ao_index = 0;

    for (auto& atom : molec.atoms) {
      for (auto& shell : atomBasis[atom.znum].shell) {
        for (uint shell_contraction = 0; shell_contraction < shell.n_contractions(); shell_contraction++) {
          // update contraction end
          contraction_end = contraction_begin + shell.contracted_gaussian.size();

          // set angular momentum
          if (shell.shell_type == SHELL::SP) {
            if (shell_contraction == 0) {
              shell_type = SHELL::S;
            } else {
              shell_type = SHELL::P;
            }
          } else {
            shell_type = shell.shell_type;
          }

          // copy alpha and norm
          for (auto kt = shell.contracted_gaussian.begin(); kt != shell.contracted_gaussian.end(); kt++) {
            auto k = std::distance(shell.contracted_gaussian.begin(), kt) + contraction_begin;
            contraction_exponents[k] = kt->first;
            contraction_coeficients[k] = kt->second[shell_contraction];
          }

          atomic_orbitals.emplace_back(
            contraction_begin,
            contraction_end,
            contraction_index,
            ao_index,
            static_cast<int>(shell_type),
            is_spherical,
            atom.pos);

          contraction_begin = contraction_end;
          contraction_index++;
          ao_index += SHELL::number_of_polynomials(shell_type, is_spherical);
          contraction++;
        }
      }
    }
  }
  atomic_orbitals.resize(n_shells);

  MPI_info::barrier();
  MPI_info::broadcast_vector_double(contraction_exponents);
  MPI_info::broadcast_vector_double(contraction_coeficients);
  MPI_info::broadcast_char((char*) atomic_orbitals.data(), n_shells * sizeof(Atomic_Orbital));
}

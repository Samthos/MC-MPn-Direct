#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../qc_mpi.h"
#include "../atom_tag_parser.h"
#include "qc_basis.h"


Basis::Basis(IOPs &iops, const Basis_Parser& basis_parser) :
  mc_num(std::max(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS])),
  qc_nbf(basis_parser.n_atomic_orbitals),
  nShells(basis_parser.n_shells),
  nPrimatives(basis_parser.n_primatives),
  lspherical(basis_parser.is_spherical),
  contraction_exp(basis_parser.contraction_exponents),
  contraction_coef(basis_parser.contraction_coeficients),
  meta_data(basis_parser.atomic_orbitals),
  contraction_amplitudes(nShells * mc_num),
  contraction_amplitudes_derivative(nShells * mc_num),
  ao_amplitudes(qc_nbf * mc_num) 
{ }


void Basis::dump(const std::string& fname) {
  std::ofstream os(fname);
  os << "\n-----------------------------------------------------------------------------------------------------------\nBasis Dump\n";
  os << "qc_nbf: " << qc_nbf << "\n";       // number basis functions
  os << "nShells: " << nShells << "\n";      // number of shells
  os << "nPrimatives: " << nPrimatives << "\n";  // number of primitives
  os << "lspherical: " << lspherical << "\n";  // true if spherical
  for (int i = 0; i < nPrimatives; ++i) {
    os << contraction_coef[i] << "\t" << contraction_exp[i] << "\n";
  }
  for (int i = 0; i < nShells; ++i) {
    os << meta_data[i].ao_begin << "\t";
    os << meta_data[i].contraction_begin << "\t";
    os << meta_data[i].contraction_end << "\t";
    os << meta_data[i].angular_momentum << "\t";
    os << meta_data[i].pos[0] << "\t";
    os << meta_data[i].pos[1] << "\t";
    os << meta_data[i].pos[2] << "\n";
  }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

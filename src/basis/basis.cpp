#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "cblas.h"
#include "../blas_calls.h"

#include "../qc_mpi.h"
#include "../atom_tag_parser.h"
#include "basis.h"


template <template <class, class> class Container, template <class> class Allocator>
Basis<Container, Allocator>::Basis(IOPs &iops, const Basis_Parser& basis_parser) :
  mc_num(std::max(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS])),
  qc_nbf(basis_parser.n_atomic_orbitals),
  nShells(basis_parser.n_shells),
  nPrimatives(basis_parser.n_primatives),
  lspherical(basis_parser.is_spherical),
  contraction_exp(basis_parser.contraction_exponents),
  contraction_coef(basis_parser.contraction_coeficients),
  contraction_amplitudes(nShells * mc_num),
  contraction_amplitudes_derivative(nShells * mc_num),
  ao_amplitudes(qc_nbf * mc_num),
  atomic_orbitals(basis_parser.atomic_orbitals)
{ }

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::host_psi_get(Wavefunction& psi, std::vector<std::array<double, 3>>& pos) {
  build_ao_amplitudes(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi.lda, qc_nbf,
      1.0,
      ao_amplitudes.data(), qc_nbf,
      psi.movecs.data(), qc_nbf,
      0.0,
      psi.psi.data(), psi.lda);
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::host_psi_get_dx(Wavefunction& psi_dx, std::vector<std::array<double, 3>>& pos) {
  // d/dx of wavefunction 
  build_ao_amplitudes_dx(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dx.lda, qc_nbf,
      1.0,
      ao_amplitudes.data(), qc_nbf,
      psi_dx.movecs.data(), qc_nbf,
      0.0,
      psi_dx.psi.data(), psi_dx.lda);
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::host_psi_get_dy(Wavefunction& psi_dy, std::vector<std::array<double, 3>>& pos) {
  // d/dy of wavefunction 
  build_ao_amplitudes_dy(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dy.lda, qc_nbf,
      1.0,
      ao_amplitudes.data(), qc_nbf,
      psi_dy.movecs.data(), qc_nbf,
      0.0,
      psi_dy.psi.data(), psi_dy.lda);
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::host_psi_get_dz(Wavefunction& psi_dz, std::vector<std::array<double, 3>>& pos) {
  // d/dz of wavefunction 
  build_ao_amplitudes_dz(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dz.lda, qc_nbf,
      1.0,
      ao_amplitudes.data(), qc_nbf,
      psi_dz.movecs.data(), qc_nbf,
      0.0,
      psi_dz.psi.data(), psi_dz.lda);
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_contractions(const std::vector<std::array<double, 3>> &pos) {
  std::fill(contraction_amplitudes.begin(), contraction_amplitudes.end(), 0.0);
  for (int walker = 0; walker < pos.size(); walker++) {
    for (auto &atomic_orbital : atomic_orbitals) {
      atomic_orbital.evaluate_contraction(
          pos[walker],
          contraction_amplitudes.data() + atomic_orbitals.size() * walker,
          contraction_exp.data(),
          contraction_coef.data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_contractions_with_derivatives(const std::vector<std::array<double, 3>>& pos) {
  std::fill(contraction_amplitudes.begin(), contraction_amplitudes.end(), 0.0);
  std::fill(contraction_amplitudes_derivative.begin(), contraction_amplitudes_derivative.end(), 0.0);
  for (int walker = 0; walker < pos.size(); walker++) {
    for (auto &atomic_orbital : atomic_orbitals) {
      atomic_orbital.evaluate_contraction_with_derivative(
          pos[walker],
          contraction_amplitudes.data() + atomic_orbitals.size() * walker,
          contraction_amplitudes_derivative.data() + atomic_orbitals.size() * walker,
          contraction_exp.data(),
          contraction_coef.data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes(const std::vector<std::array<double, 3>> &pos) {
  double x[3];
  for (int walker = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++) {
      atomic_orbitals[shell].evaluate_ao(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          pos[walker].data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dx(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker].data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dy(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker].data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dz(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker].data());
    }
  }
}

template <template <class, class> class Container, template <class> class Allocator>
void Basis<Container, Allocator>::dump(const std::string& fname) {
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
    os << atomic_orbitals[i].ao_index << "\t";
    os << atomic_orbitals[i].contraction_begin << "\t";
    os << atomic_orbitals[i].contraction_end << "\t";
    os << atomic_orbitals[i].angular_momentum << "\t";
    os << atomic_orbitals[i].pos[0] << "\t";
    os << atomic_orbitals[i].pos[1] << "\t";
    os << atomic_orbitals[i].pos[2] << "\n";
  }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

// Copyright 2017

#ifndef QC_BASIS_H_
#define QC_BASIS_H_

#include <iostream>
#include <cmath>
#include "../qc_geom.h"
#include "../qc_input.h"
#include "../qc_mpi.h"
#include "../el_pair.h"

namespace SHELL {
  enum Shell_Type {
    SP=-1, S, P, D, F, G, H
  };
  Shell_Type string_to_shell_type(const std::string& str);
  struct Shell {
    Shell_Type shell_type;
    std::vector<std::pair<double, std::vector<double>>> contracted_gaussian;
    size_t n_contractions() const {
      return contracted_gaussian.front().second.size();
    }
  };
}

struct AtomBasis {
  AtomBasis() { atomCharge = -1; }
  std::string basisName;
  std::string basisType;
  std::string atomName;
  std::vector<SHELL::Shell> shell;
  int atomCharge;
};

struct BasisMetaData{
  int angular_momentum;
  int contraction_begin;
  int contraction_end;
  int ao_begin;
  double pos[3];
};

struct BasisData {
  double *contraction_exp;
  double *contraction_coef;
  double *ao_amplitudes;  // stores AO amplidutes
  double *nw_co;          // obital coefs from nwchem

  double *psi1, *psi2, *occ1, *occ2, *vir1, *vir2;
  double *psiTau1, *psiTau2, *occTau1, *occTau2, *virTau1, *virTau2;

  BasisMetaData *meta_data;
};

class Basis {
  /*
   * Engine to compute molecular orbitals assuming
   * WARNING: Be extremely careful with this class. It manages memory explicitly.
   *
   * TODO
   *  -split of implement cartesian orbitals
   *  -specialize for GPU/CPU usage
   */
 public:
  Basis(IOPs &, MPI_info &, Molec &);
  ~Basis();
  Basis(const Basis&);
  Basis &operator=(Basis);
  friend void swap(Basis&, Basis&);

  // get psi vals
  void host_psi_get(Electron_Pair_List* el_pair);
  void host_cgs_get(const std::array<double, 3>&, int);
  void device_psi_get(double *, double *, double *, double *, double *, double *, double *, int);

  // read write
  void gpu_alloc(int, Molec &);
  void gpu_free();

  // basis set info
  int mc_pair_num;
  int iocc1;        // index first occupied orbital to be used
  int iocc2;        // index of HOMO+1
  int ivir1;        // index of LUMO
  int ivir2;        // index of last virtual to be used + 1
  int qc_nbf;       // number basis functions
  int nShells;      // number of shells
  int nPrimatives;  // number of primitives
  bool lspherical;  // true if spherical

  // from nwchem
  int nw_nbf;  // number of basis functions
  int nw_nmo;  // number of basis molecular orbital in basis set i
  double *nw_en;  // orbital energies from nwchem

  BasisData h_basis, d_basis;

 private:
  static constexpr double cf[] = {sqrt(2.5) * 0.5,
                                  sqrt(2.5) * 1.5,
                                  sqrt(15.0),
                                  sqrt(1.5) * 0.5,
                                  sqrt(6.0),
                                  1.5,
                                  sqrt(15.0) * 0.5};
  static constexpr double cg[] = {2.9580398915498085,
                                  6.2749501990055672,
                                  2.0916500663351894,
                                  1.1180339887498949,
                                  6.7082039324993694,
                                  2.3717082451262845,
                                  3.1622776601683795,
                                  0.55901699437494745,
                                  3.3541019662496847,
                                  0.73950997288745213,
                                  4.4370598373247132};

  void read(IOPs &, MPI_info &, Molec &);
  void read_new(IOPs &, MPI_info &, Molec &);
  void nw_vectors_read(IOPs &, MPI_info &, Molec &);
  void normalize();
  void normalize_sperical_atom_basis(std::vector<AtomBasis>&);
  void normalize_cartesian_atom_basis(std::vector<AtomBasis>&);
  void dump(const std::string&);
};
#endif  // QC_BASIS_H_

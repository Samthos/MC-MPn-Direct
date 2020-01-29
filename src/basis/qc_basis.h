// Copyright 2017

#ifndef QC_BASIS_H_
#define QC_BASIS_H_

#include <iostream>
#include <cmath>
#include "../qc_geom.h"
#include "../qc_input.h"
#include "../qc_mpi.h"
#include "../electron_pair_list.h"

class Wavefunction {
 public:
  Wavefunction(const size_t electrons_, int io1, int io2, int iv1, int iv2) :
    iocc1(io1),
    iocc2(io2),
    ivir1(iv1),
    ivir2(iv2),
    electrons(electrons_),
    lda(ivir2),
    psi(lda * electrons, 0.0),
    psiTau(lda * electrons, 0.0)
  {}

  const double *data() const {
    return psi.data();
  }
  const double *occ() const {
    return psi.data() + iocc1;
  }
  const double *vir() const {
    return psi.data() + ivir1;
  }
  double *dataTau() {
    return psiTau.data();
  }
  double *occTau() {
    return psiTau.data() + iocc1;
  }
  double *virTau() {
    return psiTau.data() + ivir1;
  }

  size_t iocc1;
  size_t iocc2;
  size_t ivir1;
  size_t ivir2;
  size_t number_of_molecuar_orbitals;

  size_t electrons;

  size_t lda;
  size_t rows;
  size_t col;
  // type for row/col major

  std::vector<double> psi;
  std::vector<double> psiTau;

 private:
};

namespace Cartesian_Poly {
  enum Cartesian_P {
    X = 0, 
    Y,
    Z
  };
  enum Cartesian_D {
    XX = 0,
    XY,
    XZ,
    YY,
    YZ,
    ZZ
  };
  enum Cartesian_F {
    XXX = 0,
    XXY,
    XXZ,
    XYY,
    XYZ,
    XZZ,
    YYY,
    YYZ,
    YZZ,
    ZZZ
  };
  enum Cartesian_G {
    XXXX = 0,
    XXXY,
    XXXZ,
    XXYY,
    XXYZ,
    XXZZ,
    XYYY,
    XYYZ,
    XYZZ,
    XZZZ,
    YYYY,
    YYYZ,
    YYZZ,
    YZZZ,
    ZZZZ
  };
  enum Cartesian_H {
    XXXXX = 0,
    XXXXY,
    XXXXZ,
    XXXYY,
    XXXYZ,
    XXXZZ,
    XXYYY,
    XXYYZ,
    XXYZZ,
    XXZZZ,
    XYYYY,
    XYYYZ,
    XYYZZ,
    XYZZZ,
    XZZZZ,
    YYYYY,
    YYYYZ,
    YYYZZ,
    YYZZZ,
    YZZZZ,
    ZZZZZ,
  };
}

namespace SHELL {
  enum Shell_Type {
    SP=-1, S, P, D, F, G, H
  };
  Shell_Type string_to_shell_type(const std::string& str);
  int number_of_polynomials(Shell_Type, bool spherical);
  int number_of_polynomials(int, bool spherical);
  int number_of_spherical_polynomials(Shell_Type st);
  int number_of_cartesian_polynomials(Shell_Type st);
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
  double *contraction_amplitudes;  // stores contraction amplidutes
  double *contraction_amplitudes_derivative;  // stores contraction amplidutes
  double *nw_co;          // obital coefs from nwchem
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
  void host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_with_derivatives(Wavefunction&, Wavefunction&, Wavefunction&, Wavefunction&, std::vector<std::array<double, 3>>&);
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
  void read(IOPs &, MPI_info &, Molec &);
  void nw_vectors_read(IOPs &, MPI_info &, Molec &);

  void build_contractions(const std::vector<std::array<double, 3>>&);
  void build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&);

  void build_ao_amplitudes(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&);

  static void normalize_atom_basis(std::vector<AtomBasis>&);
  static void normalize_sp(SHELL::Shell& shell);
  static void normalize_s(SHELL::Shell& shell);
  static void normalize_p(SHELL::Shell& shell);
  static void normalize_d(SHELL::Shell& shell);
  static void normalize_f(SHELL::Shell& shell);
  static void normalize_g(SHELL::Shell& shell);

  static void evaluate_spherical_d_shell(double*, double*);
  static void evaluate_spherical_f_shell(double*, double*);
  static void evaluate_spherical_g_shell(double*, double*);

  static void evaluate_s(double*, const double&, const double&, const double&, const double&);
  static void evaluate_p(double*, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_d(double*, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_f(double*, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_g(double*, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_h(double*, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_d(double*, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_f(double*, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_g(double*, const double&, const double&, const double&, const double&);

  static void evaluate_s_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_p_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_d_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_f_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_g_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_h_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_d_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_f_dx(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_g_dx(double*, const double&, const double&, const double&, const double&, const double&);

  static void evaluate_s_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_p_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_d_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_f_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_g_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_h_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_d_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_f_dy(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_g_dy(double*, const double&, const double&, const double&, const double&, const double&);

  static void evaluate_s_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_p_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_d_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_f_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_g_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_cartesian_h_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_d_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_f_dz(double*, const double&, const double&, const double&, const double&, const double&);
  static void evaluate_spherical_g_dz(double*, const double&, const double&, const double&, const double&, const double&);

  void dump(const std::string&);
};
#endif  // QC_BASIS_H_

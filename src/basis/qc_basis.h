#ifndef QC_BASIS_H_
#define QC_BASIS_H_

#include <iostream>
#include <cmath>
#include "../qc_geom.h"
#include "../qc_input.h"
#include "../qc_mpi.h"

#include "atomic_orbital.h"
#include "wavefunction.h"
#include "shell.h"
#include "atom_basis.h"
#include "basis_parser.h"

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
  Basis(IOPs&, const Basis_Parser&);

  // get psi vals
  void host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dx(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dy(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dz(Wavefunction&, std::vector<std::array<double, 3>>&);

  void build_contractions(const std::vector<std::array<double, 3>>&);
  void build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&);

  // basis set info
  int mc_num;
  int qc_nbf;       // number basis functions
  int nShells;      // number of shells
  int nPrimatives;  // number of primitives
  bool lspherical;  // true if spherical

  std::vector<double> contraction_exp;                    // dense vector of contraction exponents. Size if total number of primatives
  std::vector<double> contraction_coef;                   // dense vector of contraction coeficients. Size if total number of primatives
  std::vector<Atomic_Orbital> atomic_orbitals;

  std::vector<double> contraction_amplitudes;             // stores contraction amplitudes
  std::vector<double> contraction_amplitudes_derivative;  // stores contraction amplitudes
  std::vector<double> ao_amplitudes;                      // stores AO amplidutes

 private:
  void build_ao_amplitudes(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&);

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

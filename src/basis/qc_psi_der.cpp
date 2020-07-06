#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include "cblas.h"

#include "qc_basis.h"
#include "../blas_calls.h"

void Basis::full_host_psi_get_dx(
    Wavefunction& psi_dx,
    std::vector<std::array<double, 3>>& pos) {
  build_contractions_with_derivatives(pos);
  host_psi_get_dx(psi_dx, pos);
}
void Basis::full_host_psi_get_dy(
    Wavefunction& psi_dy,
    std::vector<std::array<double, 3>>& pos) {
  build_contractions_with_derivatives(pos);
  host_psi_get_dy(psi_dy, pos);
}
void Basis::full_host_psi_get_dz(
    Wavefunction& psi_dz,
    std::vector<std::array<double, 3>>& pos) {
  build_contractions_with_derivatives(pos);
  host_psi_get_dz(psi_dz, pos);
}

void Basis::host_psi_get_dx(
    Wavefunction& psi_dx,
    std::vector<std::array<double, 3>>& pos) {
  // d/dx of wavefunction 
  build_ao_amplitudes_dx(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dx.lda, qc_nbf,
      1.0,
      h_basis.ao_amplitudes, qc_nbf,
      psi_dx.movecs.data(), qc_nbf,
      0.0,
      psi_dx.psi.data(), psi_dx.lda);
}
void Basis::host_psi_get_dy(
    Wavefunction& psi_dy,
    std::vector<std::array<double, 3>>& pos) {
  // d/dy of wavefunction 
  build_ao_amplitudes_dy(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dy.lda, qc_nbf,
      1.0,
      h_basis.ao_amplitudes, qc_nbf,
      psi_dy.movecs.data(), qc_nbf,
      0.0,
      psi_dy.psi.data(), psi_dy.lda);
}
void Basis::host_psi_get_dz(
    Wavefunction& psi_dz,
    std::vector<std::array<double, 3>>& pos) {
  // d/dz of wavefunction 
  build_ao_amplitudes_dz(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi_dz.lda, qc_nbf,
      1.0,
      h_basis.ao_amplitudes, qc_nbf,
      psi_dz.movecs.data(), qc_nbf,
      0.0,
      psi_dz.psi.data(), psi_dz.lda);
}

void Basis::evaluate_s_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = x * rad_derivative; 
}
void Basis::evaluate_p_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[X] = x * ucgs[X] + dcgs;
  ao_amplitudes[Y] = x * ucgs[Y];
  ao_amplitudes[Z] = x * ucgs[Z];
}
void Basis::evaluate_cartesian_d_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[3];
  double ucgs[6];
  evaluate_p(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_d(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XX] = x * ucgs[XX] + dcgs[X]*2.0;
  ao_amplitudes[XY] = x * ucgs[XY] + dcgs[Y];
  ao_amplitudes[XZ] = x * ucgs[XZ] + dcgs[Z];
  ao_amplitudes[YY] = x * ucgs[YY];
  ao_amplitudes[YZ] = x * ucgs[YZ];
  ao_amplitudes[ZZ] = x * ucgs[ZZ];
}
void Basis::evaluate_cartesian_f_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[6];
  double ucgs[10];
  evaluate_cartesian_d(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_f(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XXX] = x * ucgs[XXX] + dcgs[XX]*3.0;
  ao_amplitudes[XXY] = x * ucgs[XXY] + dcgs[XY]*2.0;
  ao_amplitudes[XXZ] = x * ucgs[XXZ] + dcgs[XZ]*2.0;
  ao_amplitudes[XYY] = x * ucgs[XYY] + dcgs[YY];
  ao_amplitudes[XYZ] = x * ucgs[XYZ] + dcgs[YZ];
  ao_amplitudes[XZZ] = x * ucgs[XZZ] + dcgs[ZZ];
  ao_amplitudes[YYY] = x * ucgs[YYY];
  ao_amplitudes[YYZ] = x * ucgs[YYZ];
  ao_amplitudes[YZZ] = x * ucgs[YZZ];
  ao_amplitudes[ZZZ] = x * ucgs[ZZZ];
}
void Basis::evaluate_cartesian_g_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[10];
  double ucgs[15];
  evaluate_cartesian_f(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_g(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XXXX] = x * ucgs[XXXX] + dcgs[XXX]*4.0;
  ao_amplitudes[XXXY] = x * ucgs[XXXY] + dcgs[XXY]*3.0;
  ao_amplitudes[XXXZ] = x * ucgs[XXXZ] + dcgs[XXZ]*3.0;
  ao_amplitudes[XXYY] = x * ucgs[XXYY] + dcgs[XYY]*2.0;
  ao_amplitudes[XXYZ] = x * ucgs[XXYZ] + dcgs[XYZ]*2.0;
  ao_amplitudes[XXZZ] = x * ucgs[XXZZ] + dcgs[XZZ]*2.0;
  ao_amplitudes[XYYY] = x * ucgs[XYYY] + dcgs[YYY];
  ao_amplitudes[XYYZ] = x * ucgs[XYYZ] + dcgs[YYZ];
  ao_amplitudes[XYZZ] = x * ucgs[XYZZ] + dcgs[YZZ];
  ao_amplitudes[XZZZ] = x * ucgs[XZZZ] + dcgs[ZZZ];
  ao_amplitudes[YYYY] = x * ucgs[YYYY];
  ao_amplitudes[YYYZ] = x * ucgs[YYYZ];
  ao_amplitudes[YYZZ] = x * ucgs[YYZZ];
  ao_amplitudes[YZZZ] = x * ucgs[YZZZ];
  ao_amplitudes[ZZZZ] = x * ucgs[ZZZZ];
}
void Basis::evaluate_spherical_d_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_f_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_g_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}

void Basis::evaluate_s_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = y * rad_derivative;
}
void Basis::evaluate_p_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[0] = y * ucgs[X];
  ao_amplitudes[1] = y * ucgs[Y] + dcgs;
  ao_amplitudes[2] = y * ucgs[Z];
}
void Basis::evaluate_cartesian_d_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[3];
  double ucgs[6];
  evaluate_p(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_d(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XX] = y *ucgs[XX];
  ao_amplitudes[XY] = y *ucgs[XY] + dcgs[X];
  ao_amplitudes[XZ] = y *ucgs[XZ];
  ao_amplitudes[YY] = y *ucgs[YY] + dcgs[Y]*2.0;
  ao_amplitudes[YZ] = y *ucgs[YZ] + dcgs[Z];
  ao_amplitudes[ZZ] = y *ucgs[ZZ];
}
void Basis::evaluate_cartesian_f_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[6];
  double ucgs[10];
  evaluate_cartesian_d(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_f(&ucgs[0], rad_derivative, x, y, z);
  
  ao_amplitudes[XXX] = y * ucgs[XXX];
  ao_amplitudes[XXY] = y * ucgs[XXY] + dcgs[XX];
  ao_amplitudes[XXZ] = y * ucgs[XXZ];
  ao_amplitudes[XYY] = y * ucgs[XYY] + dcgs[XY]*2.0;
  ao_amplitudes[XYZ] = y * ucgs[XYZ] + dcgs[XZ];
  ao_amplitudes[XZZ] = y * ucgs[XZZ];
  ao_amplitudes[YYY] = y * ucgs[YYY] + dcgs[YY]*3.0;
  ao_amplitudes[YYZ] = y * ucgs[YYZ] + dcgs[YZ]*2.0;
  ao_amplitudes[YZZ] = y * ucgs[YZZ] + dcgs[ZZ];
  ao_amplitudes[ZZZ] = y * ucgs[ZZZ];
}
void Basis::evaluate_cartesian_g_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[10];
  double ucgs[15];
  evaluate_cartesian_f(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_g(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XXXX] = y * ucgs[XXXX];
  ao_amplitudes[XXXY] = y * ucgs[XXXY] + dcgs[XXX];
  ao_amplitudes[XXXZ] = y * ucgs[XXXZ];
  ao_amplitudes[XXYY] = y * ucgs[XXYY] + dcgs[XXY]*2.0;
  ao_amplitudes[XXYZ] = y * ucgs[XXYZ] + dcgs[XXZ];
  ao_amplitudes[XXZZ] = y * ucgs[XXZZ];
  ao_amplitudes[XYYY] = y * ucgs[XYYY] + dcgs[XYY]*3.0;
  ao_amplitudes[XYYZ] = y * ucgs[XYYZ] + dcgs[XYZ]*2.0;
  ao_amplitudes[XYZZ] = y * ucgs[XYZZ] + dcgs[XZZ];
  ao_amplitudes[XZZZ] = y * ucgs[XZZZ];
  ao_amplitudes[YYYY] = y * ucgs[YYYY] + dcgs[YYY]*4.0;
  ao_amplitudes[YYYZ] = y * ucgs[YYYZ] + dcgs[YYZ]*3.0;
  ao_amplitudes[YYZZ] = y * ucgs[YYZZ] + dcgs[YZZ]*2.0;
  ao_amplitudes[YZZZ] = y * ucgs[YZZZ] + dcgs[ZZZ];
  ao_amplitudes[ZZZZ] = y * ucgs[ZZZZ];
}
void Basis::evaluate_spherical_d_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_f_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_g_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}

void Basis::evaluate_s_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = z * rad_derivative;
}
void Basis::evaluate_p_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[X] = z * ucgs[X];
  ao_amplitudes[Y] = z * ucgs[Y];
  ao_amplitudes[Z] = z * ucgs[Z] + dcgs;
}
void Basis::evaluate_cartesian_d_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[3];
  double ucgs[6];
  evaluate_p(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_d(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XX] = z * ucgs[XX];
  ao_amplitudes[XY] = z * ucgs[XY];
  ao_amplitudes[XZ] = z * ucgs[XZ] + dcgs[X];
  ao_amplitudes[YY] = z * ucgs[YY];
  ao_amplitudes[YZ] = z * ucgs[YZ] + dcgs[Y];
  ao_amplitudes[ZZ] = z * ucgs[ZZ] + dcgs[Z]*2.0;
}
void Basis::evaluate_cartesian_f_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;
  double dcgs[6];
  double ucgs[10];
  evaluate_cartesian_d(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_f(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XXX] = z * ucgs[XXX];
  ao_amplitudes[XXY] = z * ucgs[XXY];
  ao_amplitudes[XXZ] = z * ucgs[XXZ] + dcgs[XX];
  ao_amplitudes[XYY] = z * ucgs[XYY];
  ao_amplitudes[XYZ] = z * ucgs[XYZ] + dcgs[XY];
  ao_amplitudes[XZZ] = z * ucgs[XZZ] + dcgs[XZ]*2.0;
  ao_amplitudes[YYY] = z * ucgs[YYY];
  ao_amplitudes[YYZ] = z * ucgs[YYZ] + dcgs[YY];
  ao_amplitudes[YZZ] = z * ucgs[YZZ] + dcgs[YZ]*2.0;
  ao_amplitudes[ZZZ] = z * ucgs[ZZZ] + dcgs[ZZ]*3.0;
}
void Basis::evaluate_cartesian_g_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs[10];
  double ucgs[15];
  evaluate_cartesian_f(&dcgs[0], rad, x, y, z);
  evaluate_cartesian_g(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[XXXX] = z * ucgs[XXXX];
  ao_amplitudes[XXXY] = z * ucgs[XXXY];
  ao_amplitudes[XXXZ] = z * ucgs[XXXZ] + dcgs[XXX];
  ao_amplitudes[XXYY] = z * ucgs[XXYY];
  ao_amplitudes[XXYZ] = z * ucgs[XXYZ] + dcgs[XXY];
  ao_amplitudes[XXZZ] = z * ucgs[XXZZ] + dcgs[XXZ]*2.0;
  ao_amplitudes[XYYY] = z * ucgs[XYYY];
  ao_amplitudes[XYYZ] = z * ucgs[XYYZ] + dcgs[XYY];
  ao_amplitudes[XYZZ] = z * ucgs[XYZZ] + dcgs[XYZ]*2.0;
  ao_amplitudes[XZZZ] = z * ucgs[XZZZ] + dcgs[XZZ]*3.0;
  ao_amplitudes[YYYY] = z * ucgs[YYYY];
  ao_amplitudes[YYYZ] = z * ucgs[YYYZ] + dcgs[YYY];
  ao_amplitudes[YYZZ] = z * ucgs[YYZZ] + dcgs[YYZ]*2.0;
  ao_amplitudes[YZZZ] = z * ucgs[YZZZ] + dcgs[YZZ]*3.0;
  ao_amplitudes[ZZZZ] = z * ucgs[ZZZZ] + dcgs[ZZZ]*4.0;
}
void Basis::evaluate_spherical_d_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_f_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_g_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}

void Basis::build_contractions_with_derivatives(const std::vector<std::array<double, 3>>& pos) {
  std::array<double, 3> dr{};
  std::fill(h_basis.contraction_amplitudes, h_basis.contraction_amplitudes + nShells * pos.size(), 0.0);
  std::fill(h_basis.contraction_amplitudes_derivative, h_basis.contraction_amplitudes_derivative + nShells * pos.size(), 0.0);
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());
      double r2 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
      for (auto i = h_basis.meta_data[shell].contraction_begin; i < h_basis.meta_data[shell].contraction_end; i++) {
        double alpha = h_basis.contraction_exp[i];
        double exponential = exp(-alpha * r2) * h_basis.contraction_coef[i];
        h_basis.contraction_amplitudes[index] += exponential;
        h_basis.contraction_amplitudes_derivative[index] -= 2.0 * alpha * exponential;
      }
    }
  }
}

void Basis::build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>& pos){
  std::array<double, 3> dr;
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      auto angular_momentum = h_basis.meta_data[shell].angular_momentum;
      auto ao_offset = walker * qc_nbf + h_basis.meta_data[shell].ao_begin;
      auto ao_amplitude = &h_basis.ao_amplitudes[ao_offset];
      auto contraction_amplitude = h_basis.contraction_amplitudes[index];
      auto contraction_amplitude_derivative = h_basis.contraction_amplitudes_derivative[index];
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());

      if (lspherical) {
        switch (angular_momentum) {
          case 0: evaluate_s_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_spherical_d_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_spherical_f_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_spherical_g_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      } else {
        switch (angular_momentum) {
          case 0: evaluate_s_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_cartesian_d_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_cartesian_f_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_cartesian_g_dx(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      }
    }
  }
}
void Basis::build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>& pos){
  std::array<double, 3> dr;
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {

      auto angular_momentum = h_basis.meta_data[shell].angular_momentum;
      auto ao_offset = walker * qc_nbf + h_basis.meta_data[shell].ao_begin;
      auto ao_amplitude = &h_basis.ao_amplitudes[ao_offset];
      auto contraction_amplitude = h_basis.contraction_amplitudes[index];
      auto contraction_amplitude_derivative = h_basis.contraction_amplitudes_derivative[index];
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());

      if (lspherical) {
        switch (angular_momentum) {
          case 0: evaluate_s_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_spherical_d_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_spherical_f_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_spherical_g_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      } else {
        switch (angular_momentum) {
          case 0: evaluate_s_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_cartesian_d_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_cartesian_f_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_cartesian_g_dy(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      }
    }
  }
}
void Basis::build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>& pos){
  std::array<double, 3> dr;
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {

      auto angular_momentum = h_basis.meta_data[shell].angular_momentum;
      auto ao_offset = walker * qc_nbf + h_basis.meta_data[shell].ao_begin;
      auto ao_amplitude = &h_basis.ao_amplitudes[ao_offset];
      auto contraction_amplitude = h_basis.contraction_amplitudes[index];
      auto contraction_amplitude_derivative = h_basis.contraction_amplitudes_derivative[index];
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());

      if (lspherical) {
        switch (angular_momentum) {
          case 0: evaluate_s_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_spherical_d_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_spherical_f_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_spherical_g_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      } else {
        switch (angular_momentum) {
          case 0: evaluate_s_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 2: evaluate_cartesian_d_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 3: evaluate_cartesian_f_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
          case 4: evaluate_cartesian_g_dz(ao_amplitude, contraction_amplitude, contraction_amplitude_derivative, dr[0], dr[1], dr[2]); break;
        }
      }
    }
  }
}

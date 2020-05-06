// Copyright 2017

#include <algorithm>
#include <cmath>
#include <iostream>
#include "cblas.h"

#include "qc_basis.h"
#include "../blas_calls.h"
#include "../electron_pair_list.h"


void Basis::gpu_alloc(int mc_pair_num, Molec& molec) {
}
void Basis::gpu_free() {
}

void Basis::full_host_psi_get(
    Wavefunction& psi,
    std::vector<std::array<double, 3>>& pos) {
  build_contractions(pos);
  host_psi_get(psi, pos);
}
void Basis::host_psi_get(
    Wavefunction& psi,
    std::vector<std::array<double, 3>>& pos) {
  build_ao_amplitudes(pos);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), psi.lda, qc_nbf,
      1.0,
      h_basis.ao_amplitudes, qc_nbf,
      psi.movecs.data(), qc_nbf,
      0.0,
      psi.psi.data(), psi.lda);
}

void Basis::build_contractions(const std::vector<std::array<double, 3>> &pos) {
  std::array<double, 3> dr{};
  std::fill(h_basis.contraction_amplitudes, h_basis.contraction_amplitudes + nShells * pos.size(), 0.0);
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());
      double r2 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
      for (auto i = h_basis.meta_data[shell].contraction_begin; i < h_basis.meta_data[shell].contraction_end; i++) {
        h_basis.contraction_amplitudes[index] += exp(-h_basis.contraction_exp[i] * r2) * h_basis.contraction_coef[i];
      }
    }
  }
}
void Basis::build_ao_amplitudes(const std::vector<std::array<double, 3>> &pos) {
  std::array<double, 3> dr;
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {

      auto angular_momentum = h_basis.meta_data[shell].angular_momentum;
      auto ao_offset = walker * qc_nbf + h_basis.meta_data[shell].ao_begin;
      auto ao_amplitude = &h_basis.ao_amplitudes[ao_offset];
      std::transform(pos[walker].begin(), pos[walker].end(), h_basis.meta_data[shell].pos, dr.begin(), std::minus<>());

      if (lspherical) {
        switch (angular_momentum) {
          case 0: evaluate_s(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 2: evaluate_spherical_d(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 3: evaluate_spherical_f(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 4: evaluate_spherical_g(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
        }
      } else {
        switch (angular_momentum) {
          case 0: evaluate_s(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1] ,dr[2]); break;
          case 1: evaluate_p(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 2: evaluate_cartesian_d(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 3: evaluate_cartesian_f(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
          case 4: evaluate_cartesian_g(ao_amplitude, h_basis.contraction_amplitudes[index], dr[0], dr[1], dr[2]); break;
        }
      }
    }
  }
}

void Basis::evaluate_s(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad;
}
void Basis::evaluate_p(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  ao_amplitudes[X] = rad * x;
  ao_amplitudes[Y] = rad * y;
  ao_amplitudes[Z] = rad * z;
}
void Basis::evaluate_cartesian_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  evaluate_p(ao_amplitudes, rad, x, y, z);
  ao_amplitudes[ZZ] = z * ao_amplitudes[Z];
  ao_amplitudes[YZ] = y * ao_amplitudes[Z];
  ao_amplitudes[YY] = y * ao_amplitudes[Y];
  ao_amplitudes[XZ] = x * ao_amplitudes[Z];
  ao_amplitudes[XY] = x * ao_amplitudes[Y];
  ao_amplitudes[XX] = x * ao_amplitudes[X];
}
void Basis::evaluate_cartesian_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  evaluate_cartesian_d(ao_amplitudes, rad, x, y, z);
  ao_amplitudes[ZZZ] = z * ao_amplitudes[ZZ];
  ao_amplitudes[YZZ] = y * ao_amplitudes[ZZ];
  ao_amplitudes[YYZ] = y * ao_amplitudes[YZ];
  ao_amplitudes[YYY] = y * ao_amplitudes[YY];
  ao_amplitudes[XZZ] = x * ao_amplitudes[ZZ];
  ao_amplitudes[XYZ] = x * ao_amplitudes[YZ];
  ao_amplitudes[XYY] = x * ao_amplitudes[YY];
  ao_amplitudes[XXZ] = x * ao_amplitudes[XZ];
  ao_amplitudes[XXY] = x * ao_amplitudes[XY];
  ao_amplitudes[XXX] = x * ao_amplitudes[XX];
}
void Basis::evaluate_cartesian_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  evaluate_cartesian_f(ao_amplitudes, rad, x, y, z);
  ao_amplitudes[ZZZZ] = z * ao_amplitudes[ZZZ];
  ao_amplitudes[YZZZ] = y * ao_amplitudes[ZZZ];
  ao_amplitudes[YYZZ] = y * ao_amplitudes[YZZ];
  ao_amplitudes[YYYZ] = y * ao_amplitudes[YYZ];
  ao_amplitudes[YYYY] = y * ao_amplitudes[YYY];
  ao_amplitudes[XZZZ] = x * ao_amplitudes[ZZZ];
  ao_amplitudes[XYZZ] = x * ao_amplitudes[YZZ];
  ao_amplitudes[XYYZ] = x * ao_amplitudes[YYZ];
  ao_amplitudes[XYYY] = x * ao_amplitudes[YYY];
  ao_amplitudes[XXZZ] = x * ao_amplitudes[XZZ];
  ao_amplitudes[XXYZ] = x * ao_amplitudes[XYZ];
  ao_amplitudes[XXYY] = x * ao_amplitudes[XYY];
  ao_amplitudes[XXXZ] = x * ao_amplitudes[XXZ];
  ao_amplitudes[XXXY] = x * ao_amplitudes[XXY];
  ao_amplitudes[XXXX] = x * ao_amplitudes[XXX];
}
void Basis::evaluate_cartesian_h(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  evaluate_cartesian_g(ao_amplitudes, rad, x, y, z);
  ao_amplitudes[ZZZZZ] = z * ao_amplitudes[ZZZZ];
  ao_amplitudes[YZZZZ] = y * ao_amplitudes[ZZZZ];
  ao_amplitudes[YYZZZ] = y * ao_amplitudes[YZZZ];
  ao_amplitudes[YYYZZ] = y * ao_amplitudes[YYZZ];
  ao_amplitudes[YYYYZ] = y * ao_amplitudes[YYYZ];
  ao_amplitudes[YYYYY] = y * ao_amplitudes[YYYY];
  ao_amplitudes[XZZZZ] = x * ao_amplitudes[ZZZZ];
  ao_amplitudes[XYZZZ] = x * ao_amplitudes[YZZZ];
  ao_amplitudes[XYYZZ] = x * ao_amplitudes[YYZZ];
  ao_amplitudes[XYYYZ] = x * ao_amplitudes[YYYZ];
  ao_amplitudes[XYYYY] = x * ao_amplitudes[YYYY];
  ao_amplitudes[XXZZZ] = x * ao_amplitudes[XZZZ];
  ao_amplitudes[XXYZZ] = x * ao_amplitudes[XYZZ];
  ao_amplitudes[XXYYZ] = x * ao_amplitudes[XYYZ];
  ao_amplitudes[XXYYY] = x * ao_amplitudes[XYYY];
  ao_amplitudes[XXXZZ] = x * ao_amplitudes[XXZZ];
  ao_amplitudes[XXXYZ] = x * ao_amplitudes[XXYZ];
  ao_amplitudes[XXXYY] = x * ao_amplitudes[XXYY];
  ao_amplitudes[XXXXZ] = x * ao_amplitudes[XXXZ];
  ao_amplitudes[XXXXY] = x * ao_amplitudes[XXXY];
  ao_amplitudes[XXXXX] = x * ao_amplitudes[XXXX];
}

void Basis::evaluate_spherical_d_shell(double* ao_amplitudes, double* ang) {
  using namespace Cartesian_Poly;
  constexpr double cd[] = {1.732050807568877, // sqrt(3)
                           0.86602540378443}; // 0.5 * sqrt(3)
  ao_amplitudes[0] =  cd[0] * ang[XY];
  ao_amplitudes[1] =  cd[0] * ang[YZ];
  ao_amplitudes[2] =  0.5 * (2.0 * ang[ZZ] - ang[XX] - ang[YY]);
  ao_amplitudes[3] = -cd[0] * ang[XZ];
  ao_amplitudes[4] =  cd[1] * (ang[XX] - ang[YY]);
}
void Basis::evaluate_spherical_f_shell(double* ao_amplitudes, double* ang) {
  using namespace Cartesian_Poly;
  constexpr double cf[] = {0.7905694150420949,  // sqrt(2.5) * 0.5,
                           2.3717082451262845,  // sqrt(2.5) * 1.5,
                           3.8729833462074170,  // sqrt(15.0),
                           0.6123724356957945,  // sqrt(1.5) * 0.5,
                           2.4494897427831780,  // sqrt(6.0),
                           1.5000000000000000,  // 1.5,
                           1.9364916731037085}; // sqrt(15.0) * 0.5
  ao_amplitudes[0] = cf[1] * ang[XXY] - cf[0] * ang[YYY];
  ao_amplitudes[1] = cf[2] * ang[XYZ];
  ao_amplitudes[2] = cf[4] * ang[YZZ] - cf[3] * (ang[XXY] + ang[YYY]);
  ao_amplitudes[3] = ang[ZZZ] - cf[5] * (ang[XXZ] + ang[YYZ]);
  ao_amplitudes[4] = cf[3] * (ang[XXX] + ang[XYY]) - cf[4] * ang[XZZ];
  ao_amplitudes[5] = cf[6] * (ang[XXZ] - ang[YYZ]);
  ao_amplitudes[6] = cf[1] * ang[XYY] - cf[0] * ang[XXX];
}
void Basis::evaluate_spherical_g_shell(double* ao_amplitudes, double* ang) {
  using namespace Cartesian_Poly;
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
  ao_amplitudes[0] = cg[0] * (ang[XXXY] - ang[XYYY]);
  ao_amplitudes[1] = cg[1] * ang[XXYZ] - cg[2] * ang[YYYZ];
  ao_amplitudes[2] = cg[4] * ang[XYZZ] - cg[3] * (ang[XXXY] + ang[XYYY]);
  ao_amplitudes[3] = cg[6] * ang[YZZZ] - cg[5] * ang[XXYZ] - cg[5] * ang[YYYZ];
  ao_amplitudes[4] = 0.375 * (ang[XXXX] + ang[YYYY] + 2.0 * ang[XXYY]) + ang[ZZZZ] - 3.0 * (ang[XXZZ] + ang[YYZZ]);
  ao_amplitudes[5] = cg[5] * ang[XXXZ] + cg[5] * ang[XYYZ] - cg[6] * ang[XZZZ];
  ao_amplitudes[6] = cg[7] * (ang[YYYY] - ang[XXXX]) + cg[8] * (ang[XXZZ] - ang[YYZZ]);
  ao_amplitudes[7] = cg[1] * ang[XYYZ] - cg[2] * ang[XXXZ];
  ao_amplitudes[8] = cg[9] * (ang[XXXX] + ang[YYYY]) - cg[10] * ang[XXYY];
}

void Basis::evaluate_spherical_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[6];
  evaluate_cartesian_d(&ang[0], rad, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[10];
  evaluate_cartesian_f(&ang[0], rad, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
void Basis::evaluate_spherical_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[15];
  evaluate_cartesian_g(&ang[0], rad, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}


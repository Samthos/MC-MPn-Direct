// Copyright 2017

#include <algorithm>
#include <cmath>
#include <iostream>

#include "cblas.h"
#include "../blas_calls.h"
#include "qc_basis.h"
#include "../electron_pair_list.h"


void Basis::gpu_alloc(int mc_pair_num, Molec& molec) {
}
void Basis::gpu_free() {
}

void Basis::host_psi_get(Wavefunction& psi, std::vector<std::array<double, 3>>& pos) {
  for (auto walker = 0; walker < pos.size(); ++walker) {
    host_cgs_get(pos[walker], walker);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      pos.size(), ivir2 - iocc1, nw_nbf,
      1.0,
      h_basis.ao_amplitudes, nw_nbf,
      h_basis.nw_co + iocc1 * nw_nbf, nw_nbf,
      0.0,
      psi.psi.data(), ivir2-iocc1);
}

void Basis::host_cgs_get(const std::array<double, 3> &pos, const int walker) {
  double r2, rad;
  double x, y, z;

  for (int shell = 0; shell < nShells; shell++) {
    x = pos[0] - h_basis.meta_data[shell].pos[0];
    y = pos[1] - h_basis.meta_data[shell].pos[1];
    z = pos[2] - h_basis.meta_data[shell].pos[2];
    r2 = x * x + y * y + z * z;

    rad = 0.0;
    for (auto i = h_basis.meta_data[shell].contraction_begin; i < h_basis.meta_data[shell].contraction_end; i++) {
      rad = rad + exp(-h_basis.contraction_exp[i] * r2) * h_basis.contraction_coef[i];
    }

    auto angular_momentum = h_basis.meta_data[shell].angular_momentum;
    auto ao_offset = walker * nw_nbf + h_basis.meta_data[shell].ao_begin;
    auto ao_amplitude = &h_basis.ao_amplitudes[ao_offset];
    if (lspherical) {
      switch (angular_momentum) {
        case 0: evaulate_s(ao_amplitude, rad, x, y ,z); break;
        case 1: evaulate_p(ao_amplitude, rad, x, y, z); break;
        case 2: evaulate_spherical_d(ao_amplitude, rad, x, y, z); break;
        case 3: evaulate_spherical_f(ao_amplitude, rad, x, y, z); break;
        case 4: evaulate_spherical_g(ao_amplitude, rad, x, y, z); break;
      }
    } else {
      switch (angular_momentum) {
        case 0: evaulate_s(ao_amplitude, rad, x, y ,z); break;
        case 1: evaulate_p(ao_amplitude, rad, x, y, z); break;
        case 2: evaulate_cartesian_d(ao_amplitude, rad, x, y, z); break;
        case 3: evaulate_cartesian_f(ao_amplitude, rad, x, y, z); break;
        case 4: evaulate_cartesian_g(ao_amplitude, rad, x, y, z); break;
      }
    }
  }
}
void Basis::evaulate_s(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad;
}
void Basis::evaulate_p(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad * x;
  ao_amplitudes[1] = rad * y;
  ao_amplitudes[2] = rad * z;
}
void Basis::evaulate_spherical_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  std::array<double, 5> ang;
  ang[0] = 1.732050807568877 * x * y;            // sqrt(3)
  ang[1] = 1.732050807568877 * y * z;            // sqrt(3)
  ang[2] = 0.5 * (2.0 * z * z - x * x - y * y);  //
  ang[3] = -1.732050807568877 * x * z;           // sqrt(3)
  ang[4] = 0.86602540378443 * (x * x - y * y);   // sqrt(3) / 2

  ao_amplitudes[0] = rad * ang[0];
  ao_amplitudes[1] = rad * ang[1];
  ao_amplitudes[2] = rad * ang[2];
  ao_amplitudes[3] = rad * ang[3];
  ao_amplitudes[4] = rad * ang[4];
}
void Basis::evaulate_spherical_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  std::array<double, 7> ang;
  ang[0] = y * (cf[1] * x * x - cf[0] * y * y);  // xxy, yyy
  ang[1] = cf[2] * x * y * z;
  ang[2] = y * (cf[4] * z * z - cf[3] * (x * x + y * y));
  ang[3] = z * (z * z - cf[5] * (x * x + y * y));
  ang[4] = -x * (cf[4] * z * z - cf[3] * (x * x + y * y));
  ang[5] = z * cf[6] * (x * x - y * y);
  ang[6] = x * (cf[1] * y * y - cf[0] * x * x);

  ao_amplitudes[0] = rad * ang[0];
  ao_amplitudes[1] = rad * ang[1];
  ao_amplitudes[2] = rad * ang[2];
  ao_amplitudes[3] = rad * ang[3];
  ao_amplitudes[4] = rad * ang[4];
  ao_amplitudes[5] = rad * ang[5];
  ao_amplitudes[6] = rad * ang[6];
}
void Basis::evaulate_spherical_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  std::array<double, 9> ang;
  // m = -4
  ang[0] = cg[0] * (x * x * x * y - x * y * y * y);  // xxxy, xyyy
  // m = -3
  ang[1] = y * z * (cg[1] * x * x - cg[2] * y * y);  // (2, 1, 1) (0, 3, 1)
  // m = -2
  ang[2] = x * y * cg[3] * (-x * x - y * y) + cg[4] * x * y * z * z;
  // m = -1
  ang[3] = -cg[5] * x * x * y * z - cg[5] * y * y * y * z + cg[6] * y * z * z * z;
  // m = 0
  ang[4] = 0.375 * (x * x * x * x + y * y * y * y + 2.0 * x * x * y * y) + z * z * z * z
      - 3.0 * z * z * (x * x + y * y);
  // m = 1
  ang[5] = cg[5] * x * x * x * z + cg[5] * x * y * y * z - cg[6] * x * z * z * z;
  // m = 2
  ang[6] = cg[7] * (y * y * y * y - x * x * x * x) + cg[8] * z * z * (x * x - y * y);
  // m = 3
  ang[7] = x * z * (cg[1] * y * y - cg[2] * x * x);  // (1, 2, 1) (3, 0, 1)
  // m = 4
  ang[8] = cg[9] * (x * x * x * x + y * y * y * y) - cg[10] * x * x * y * y;

  ao_amplitudes[0] = rad * ang[0];
  ao_amplitudes[1] = rad * ang[1];
  ao_amplitudes[2] = rad * ang[2];
  ao_amplitudes[3] = rad * ang[3];
  ao_amplitudes[4] = rad * ang[4];
  ao_amplitudes[5] = rad * ang[5];
  ao_amplitudes[6] = rad * ang[6];
  ao_amplitudes[7] = rad * ang[7];
  ao_amplitudes[8] = rad * ang[8];
}
void Basis::evaulate_cartesian_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad * x*x;
  ao_amplitudes[1] = rad * x*y;
  ao_amplitudes[2] = rad * x*z;
  ao_amplitudes[3] = rad * y*y;
  ao_amplitudes[4] = rad * y*z;
  ao_amplitudes[5] = rad * z*z;
}
void Basis::evaulate_cartesian_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad * x*x*x;
  ao_amplitudes[1] = rad * x*x*y;
  ao_amplitudes[2] = rad * x*x*z;
  ao_amplitudes[3] = rad * x*y*y;
  ao_amplitudes[4] = rad * x*y*z;
  ao_amplitudes[5] = rad * x*z*z;
  ao_amplitudes[6] = rad * y*y*y;
  ao_amplitudes[7] = rad * y*y*z;
  ao_amplitudes[8] = rad * y*z*z;
  ao_amplitudes[9] = rad * z*z*z;
}
void Basis::evaulate_cartesian_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[ 0] = rad * x*x*x*x;
  ao_amplitudes[ 1] = rad * x*x*x*y;
  ao_amplitudes[ 2] = rad * x*x*x*z;
  ao_amplitudes[ 3] = rad * x*x*y*y;
  ao_amplitudes[ 4] = rad * x*x*y*z;
  ao_amplitudes[ 5] = rad * x*x*z*z;
  ao_amplitudes[ 6] = rad * x*y*y*y;
  ao_amplitudes[ 7] = rad * x*y*y*z;
  ao_amplitudes[ 8] = rad * x*y*z*z;
  ao_amplitudes[ 9] = rad * x*z*z*z;
  ao_amplitudes[10] = rad * y*y*y*y;
  ao_amplitudes[11] = rad * y*y*y*z;
  ao_amplitudes[12] = rad * y*y*z*z;
  ao_amplitudes[13] = rad * y*z*z*z;
  ao_amplitudes[14] = rad * z*z*z*z;
}

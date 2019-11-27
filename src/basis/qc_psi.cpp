// Copyright 2017

#include <algorithm>
#include <cmath>
#include <iostream>

#include "cblas.h"
#include "../blas_calls.h"
#include "qc_basis.h"
#include "../el_pair.h"


void Basis::gpu_alloc(int mc_pair_num, Molec& molec) {
}
void Basis::gpu_free() {
}

void Basis::host_psi_get(Electron_Pair_List* el_pair) {
  //for (auto &walker : el_pair) {
  for (auto walker = 0; walker < el_pair->size(); ++walker) {
    host_cgs_get(el_pair->get(walker).pos1, walker);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      el_pair->size(), ivir2 - iocc1, nw_nbf,
      1.0,
      h_basis.ao_amplitudes, nw_nbf,
      h_basis.nw_co + iocc1 * nw_nbf, nw_nbf,
      0.0,
      h_basis.psi1, ivir2-iocc1);

  for (auto walker = 0; walker < el_pair->size(); ++walker) {
    host_cgs_get(el_pair->get(walker).pos2, walker);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      el_pair->size(), ivir2 - iocc1, nw_nbf,
      1.0,
      h_basis.ao_amplitudes, nw_nbf,
      h_basis.nw_co + iocc1 * nw_nbf, nw_nbf,
      0.0,
      h_basis.psi2, ivir2-iocc1);
}

void Basis::host_cgs_get(const std::array<double, 3> &pos, const int walker) {
  int ic, iam;
  double r2, rad;
  double x, y, z;
  double ang[15];

  for (int shell = 0; shell < nShells; shell++) {
    iam = h_basis.meta_data[shell].angular_momentum;

    x = pos[0] - h_basis.meta_data[shell].pos[0];
    y = pos[1] - h_basis.meta_data[shell].pos[1];
    z = pos[2] - h_basis.meta_data[shell].pos[2];
    r2 = x * x + y * y + z * z;

    rad = 0.0;
    for (auto i = h_basis.meta_data[shell].contraction_begin; i < h_basis.meta_data[shell].contraction_end; i++) {
      rad = rad + exp(-h_basis.contraction_exp[i] * r2) * h_basis.contraction_coef[i];
    }

    if (lspherical) {
      ic = walker * nw_nbf + h_basis.meta_data[shell].ao_begin;
      switch (iam) {
        case 0:
          h_basis.ao_amplitudes[ic + 0] = rad;
          break;
        case 1:
          h_basis.ao_amplitudes[ic + 0] = rad * x;
          h_basis.ao_amplitudes[ic + 1] = rad * y;
          h_basis.ao_amplitudes[ic + 2] = rad * z;
          break;
        case 2:
          ang[0] = 1.732050807568877 * x * y;            // sqrt(3)
          ang[1] = 1.732050807568877 * y * z;            // sqrt(3)
          ang[2] = 0.5 * (2.0 * z * z - x * x - y * y);  //
          ang[3] = -1.732050807568877 * x * z;           // sqrt(3)
          ang[4] = 0.86602540378443 * (x * x - y * y);   // sqrt(3) / 2

          h_basis.ao_amplitudes[ic + 0] = rad * ang[0];
          h_basis.ao_amplitudes[ic + 1] = rad * ang[1];
          h_basis.ao_amplitudes[ic + 2] = rad * ang[2];
          h_basis.ao_amplitudes[ic + 3] = rad * ang[3];
          h_basis.ao_amplitudes[ic + 4] = rad * ang[4];
          break;
        case 3:
          ang[0] = y * (cf[1] * x * x - cf[0] * y * y);  // xxy, yyy
          ang[1] = cf[2] * x * y * z;
          ang[2] = y * (cf[4] * z * z - cf[3] * (x * x + y * y));
          ang[3] = z * (z * z - cf[5] * (x * x + y * y));
          ang[4] = -x * (cf[4] * z * z - cf[3] * (x * x + y * y));
          ang[5] = z * cf[6] * (x * x - y * y);
          ang[6] = x * (cf[1] * y * y - cf[0] * x * x);

          h_basis.ao_amplitudes[ic + 0] = rad * ang[0];
          h_basis.ao_amplitudes[ic + 1] = rad * ang[1];
          h_basis.ao_amplitudes[ic + 2] = rad * ang[2];
          h_basis.ao_amplitudes[ic + 3] = rad * ang[3];
          h_basis.ao_amplitudes[ic + 4] = rad * ang[4];
          h_basis.ao_amplitudes[ic + 5] = rad * ang[5];
          h_basis.ao_amplitudes[ic + 6] = rad * ang[6];
          break;
        case 4:
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

          h_basis.ao_amplitudes[ic + 0] = rad * ang[0];
          h_basis.ao_amplitudes[ic + 1] = rad * ang[1];
          h_basis.ao_amplitudes[ic + 2] = rad * ang[2];
          h_basis.ao_amplitudes[ic + 3] = rad * ang[3];
          h_basis.ao_amplitudes[ic + 4] = rad * ang[4];
          h_basis.ao_amplitudes[ic + 5] = rad * ang[5];
          h_basis.ao_amplitudes[ic + 6] = rad * ang[6];
          h_basis.ao_amplitudes[ic + 7] = rad * ang[7];
          h_basis.ao_amplitudes[ic + 8] = rad * ang[8];
          break;
      }
    } else {
      ic = walker * nw_nbf + h_basis.meta_data[shell].ao_begin;
      switch (iam) {
        case 0:
          h_basis.ao_amplitudes[ic+0] = rad;
          break;
        case 1:
          h_basis.ao_amplitudes[ic+0] = rad * x;
          h_basis.ao_amplitudes[ic+1] = rad * y;
          h_basis.ao_amplitudes[ic+2] = rad * z;
          break;
        case 2:
          h_basis.ao_amplitudes[ic+0] = rad * x*x;
          h_basis.ao_amplitudes[ic+1] = rad * x*y;  // sqrt(3)
          h_basis.ao_amplitudes[ic+2] = rad * x*z;  // sqrt(3)
          h_basis.ao_amplitudes[ic+3] = rad * y*y;
          h_basis.ao_amplitudes[ic+4] = rad * y*z;  // sqrt(3)
          h_basis.ao_amplitudes[ic+5] = rad * z*z;
          break;
        case 3:
          h_basis.ao_amplitudes[ic+0] = rad * x*x*x; // fxxx
          h_basis.ao_amplitudes[ic+1] = rad * x*x*y; // sqrt_5*x*x*y  // fxxy
          h_basis.ao_amplitudes[ic+2] = rad * x*x*z; // sqrt_5*x*x*z  // fxxz
          h_basis.ao_amplitudes[ic+3] = rad * x*y*y; // sqrt_5*x*y*y  // fxyy
          h_basis.ao_amplitudes[ic+4] = rad * x*y*z; // cf(3)*x*y*z  // fxyz
          h_basis.ao_amplitudes[ic+5] = rad * x*z*z; // sqrt_5*x*z*z  // fxzz
          h_basis.ao_amplitudes[ic+6] = rad * y*y*y; // fyyy
          h_basis.ao_amplitudes[ic+7] = rad * y*y*z; // sqrt_5*y*y*z  // fyyz
          h_basis.ao_amplitudes[ic+8] = rad * y*z*z; // sqrt_5*y*z*z  // fyzz
          h_basis.ao_amplitudes[ic+9] = rad * z*z*z; // fzzz
          break ;
        case 4:
          h_basis.ao_amplitudes[ic+ 0] = rad * x*x*x*x;
          h_basis.ao_amplitudes[ic+ 1] = rad * x*x*x*y;
          h_basis.ao_amplitudes[ic+ 2] = rad * x*x*x*z;
          h_basis.ao_amplitudes[ic+ 3] = rad * x*x*y*y;
          h_basis.ao_amplitudes[ic+ 4] = rad * x*x*y*z;
          h_basis.ao_amplitudes[ic+ 5] = rad * x*x*z*z;
          h_basis.ao_amplitudes[ic+ 6] = rad * x*y*y*y;
          h_basis.ao_amplitudes[ic+ 7] = rad * x*y*y*z;
          h_basis.ao_amplitudes[ic+ 8] = rad * x*y*z*z;
          h_basis.ao_amplitudes[ic+ 9] = rad * x*z*z*z;
          h_basis.ao_amplitudes[ic+10] = rad * y*y*y*y;
          h_basis.ao_amplitudes[ic+11] = rad * y*y*y*z;
          h_basis.ao_amplitudes[ic+12] = rad * y*y*z*z;
          h_basis.ao_amplitudes[ic+13] = rad * y*z*z*z;
          h_basis.ao_amplitudes[ic+14] = rad * z*z*z*z;
          break;
      }
    }
  }
}

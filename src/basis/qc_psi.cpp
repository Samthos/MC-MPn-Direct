// Copyright 2017

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../blas_calls.h"
#include "qc_basis.h"
#include "../el_pair.h"

void Basis::gpu_alloc(int mc_pair_num, Molec& molec) {
}
void Basis::gpu_free() {
}

void Basis::host_psi_get(std::vector<el_pair_typ>& el_pair) {
  for (auto &walker : el_pair) {
    host_cgs_get(walker.pos1);

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ivir2-iocc1, nw_nbf,
                1.0, h_basis.nw_co + iocc1*nw_nbf, ivir2,
                h_basis.ao_amplitudes, 1,
                0.0, walker.psi1.data()+iocc1, 1);

    host_cgs_get(walker.pos2);

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                ivir2-iocc1, nw_nbf,
                1.0, h_basis.nw_co + iocc1*nw_nbf, ivir2,
                h_basis.ao_amplitudes, 1,
                0.0, walker.psi2.data()+iocc1, 1);
  }

  for (auto ip = 0; ip < mc_pair_num; ip++) {
    for (auto am = iocc1; am < iocc2; am++) {
      h_basis.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair[ip].psi1[am];
      h_basis.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair[ip].psi2[am];
    }
    for (auto am = ivir1; am < ivir2; am++) {
      h_basis.psi1[(am - iocc1) * mc_pair_num + ip] = el_pair[ip].psi1[am];
      h_basis.psi2[(am - iocc1) * mc_pair_num + ip] = el_pair[ip].psi2[am];
    }
  }
}

void Basis::host_cgs_get(std::array<double, 3> &pos) {
  int ic, iat, iam;
  double r2, rad;
  double x, y, z;
  double ang[15];

  for (int shell = 0; shell < nShells; shell++) {
    iam = h_basis.meta_data[shell].angular_moment;

    x = pos[0] - h_basis.meta_data[shell].pos[0];
    y = pos[1] - h_basis.meta_data[shell].pos[1];
    z = pos[2] - h_basis.meta_data[shell].pos[2];
    r2 = x * x + y * y + z * z;

    rad = 0.0;
    for (auto i = h_basis.meta_data[shell].contraction_begin; i < h_basis.meta_data[shell].contraction_end; i++) {
      rad = rad + exp(-h_basis.contraction_exp[i] * r2) * h_basis.contraction_coef[i];
    }

    if (lspherical) {
      ic = h_basis.meta_data[shell].ao_begin;
      switch (iam) {
        case 0:
          h_basis.ao_amplitudes[ic + 0] = rad;
          break;
        case -1:
          h_basis.ao_amplitudes[ic + 0] = rad;
          h_basis.ao_amplitudes[ic + 1] = rad * x;
          h_basis.ao_amplitudes[ic + 2] = rad * y;
          h_basis.ao_amplitudes[ic + 3] = rad * z;
          break;
        case 1:
          h_basis.ao_amplitudes[ic + 0] = rad * x;
          h_basis.ao_amplitudes[ic + 1] = rad * y;
          h_basis.ao_amplitudes[ic + 2] = rad * z;
          break;
        case 2:
          ang[0] = 1.732050807568877 * x * y;            // dxy
          ang[1] = 1.732050807568877 * y * z;            // dyz
          ang[2] = 0.5 * (2.0 * z * z - x * x - y * y);  // dxx, dyy, dzz
          ang[3] = -1.732050807568877 * x * z;           // dxz
          ang[4] = 0.86602540378443 * (x * x - y * y);   // dxx, dyy

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
    }
    /*
    else {
      // cartesian GTO
      // print *, 'ish am ic ', is, ishl%am, ishl%ao_amplitudes

      ic = ishl%ao_amplitudes - 1
      if (ishl%am == 0) {
        ao_amplitudes(ic+1) = rad(1)
      } else if (ishl%am == -1) {
        ao_amplitudes(ic+1) = rad(1)
        ao_amplitudes(ic+2) = rad(2)*x
        ao_amplitudes(ic+3) = rad(2)*y
        ao_amplitudes(ic+4) = rad(2)*z
      } else if (ishl%am == 1) {
        ao_amplitudes(ic+1) = rad(1)*x
        ao_amplitudes(ic+2) = rad(1)*y
        ao_amplitudes(ic+3) = rad(1)*z
      } else if (ishl%am == 2) {
        ang(1) = x*x  // dxx
        ang(2) = x*y  // cd(1)*x*y  // dxy
        ang(3) = x*z  // cd(1)*x*z  // dxz
        ang(4) = y*y  // dyy
        ang(5) = y*z  // cd(1)*y*z  // dyz
        ang(6) = z*z  // dzz

        ao_amplitudes(ic+1) = rad(1)*ang(1)
        ao_amplitudes(ic+2) = rad(1)*ang(2)
        ao_amplitudes(ic+3) = rad(1)*ang(3)
        ao_amplitudes(ic+4) = rad(1)*ang(4)
        ao_amplitudes(ic+5) = rad(1)*ang(5)
        ao_amplitudes(ic+6) = rad(1)*ang(6)
      } else if (ishl%am == 3) {
        ang(1) = x*x*x  // fxxx
        ang(2) = x*x*y  // sqrt_5*x*x*y  // fxxy
        ang(3) = x*x*z  // sqrt_5*x*x*z  // fxxz
        ang(4) = x*y*y  // sqrt_5*x*y*y  // fxyy
        ang(5) = x*y*z  // cf(3)*x*y*z  // fxyz
        ang(6) = x*z*z  // sqrt_5*x*z*z  // fxzz
        ang(7) = y*y*y  // fyyy
        ang(8) = y*y*z  // sqrt_5*y*y*z  // fyyz
        ang(9) = y*z*z  // sqrt_5*y*z*z  // fyzz
        ang(10) = z*z*z  // fzzz

        ao_amplitudes(ic+1) = rad(1)*ang(1)
        ao_amplitudes(ic+2) = rad(1)*ang(2)
        ao_amplitudes(ic+3) = rad(1)*ang(3)
        ao_amplitudes(ic+4) = rad(1)*ang(4)
        ao_amplitudes(ic+5) = rad(1)*ang(5)
        ao_amplitudes(ic+6) = rad(1)*ang(6)
        ao_amplitudes(ic+7) = rad(1)*ang(7)
        ao_amplitudes(ic+8) = rad(1)*ang(8)
        ao_amplitudes(ic+9) = rad(1)*ang(9)
        ao_amplitudes(ic+10) = rad(1)*ang(10)
      } else if (ishl%am == 4) {
        ang(1) = x*x*x*x  // (4, 0, 0)
        ang(2) = x*x*x*y  // (3, 1, 0)
        ang(3) = x*x*x*z  // (3, 0, 1)
        ang(4) = x*x*y*y  // (2, 2, 0)
        ang(5) = x*x*y*z  // (2, 1, 1)
        ang(6) = x*x*z*z  // (2, 0, 2)
        ang(7) = x*y*y*y  // (1, 3, 0)
        ang(8) = x*y*y*z  // (1, 2, 1)
        ang(9) = x*y*z*z  // (1, 1, 2)
        ang(10)= x*z*z*z  // (1, 0, 3)

        ang(11)= y*y*y*y  // (0, 4, 0)
        ang(12)= y*y*y*z  // (0, 3, 1)
        ang(13)= y*y*z*z  // (0, 2, 2)
        ang(14)= y*z*z*z  // (0, 1, 3)
        ang(15)= z*z*z*z  // (0, 0, 4)

        ao_amplitudes(ic+1) = rad(1)*ang(1)
        ao_amplitudes(ic+2) = rad(1)*ang(2)
        ao_amplitudes(ic+3) = rad(1)*ang(3)
        ao_amplitudes(ic+4) = rad(1)*ang(4)
        ao_amplitudes(ic+5) = rad(1)*ang(5)
        ao_amplitudes(ic+6) = rad(1)*ang(6)
        ao_amplitudes(ic+7) = rad(1)*ang(7)
        ao_amplitudes(ic+8) = rad(1)*ang(8)
        ao_amplitudes(ic+9) = rad(1)*ang(9)
        ao_amplitudes(ic+10) = rad(1)*ang(10)
        ao_amplitudes(ic+11) = rad(1)*ang(11)
        ao_amplitudes(ic+12) = rad(1)*ang(12)
        ao_amplitudes(ic+13) = rad(1)*ang(13)
        ao_amplitudes(ic+14) = rad(1)*ang(14)
        ao_amplitudes(ic+15) = rad(1)*ang(15)
      }
    }
    */
    // std::cerr << std::endl;
  }
}

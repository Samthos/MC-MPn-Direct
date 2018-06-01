// Copyright 2017

#include <algorithm>
#include <cmath>
#include <iostream>

#include "qc_basis.h"

void Basis::gpu_alloc(int mc_pair_num, Molec& molec) {
}
void Basis::gpu_free() {
}

void Basis::host_psi_get(double* pos, double* psi, Molec& molec) {
  int i, j, index;
  double psii;

  host_cgs_get(pos, molec);

  index = iocc1 * nw_nbf;
  for (i = iocc1; i < ivir2; i++) {
    psii = 0.0;
    for (j = 0; j < nw_nbf; j++) {
      psii = psii + h_basis.icgs[j] * h_basis.nw_co[index];
      index++;
    }
    psi[i] = psii;
  }
}

void Basis::host_cgs_get(double* pos, Molec& molec) {
  int ic, iat, iam, i, j;
  double r2, rad;
  double pos_i[3], dr[3], x, y, z;

  for (i = 0; i < qc_nshl; i++) {
    iat = h_basis.at[i];
    pos_i[0] = molec.atom[iat].pos[0];
    pos_i[1] = molec.atom[iat].pos[1];
    pos_i[2] = molec.atom[iat].pos[2];

    iam = h_basis.am[i];

    //    std::cerr << i << "\t" << qc_nshl << "\t" << iat << "\t" << iam << "\t" ;
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos_i[0] << "\t";
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos_i[1] << "\t";
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos_i[2] << "\t";
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos[0]   << "\t";
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos[1]   << "\t";
    //    std::cerr << std::setw(7) << std::setprecision(4) << pos[2]   << "\t";

    dr[0] = pos[0] - pos_i[0];
    dr[1] = pos[1] - pos_i[1];
    dr[2] = pos[2] - pos_i[2];
    r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

    //  std::cerr << std::setw(7) << std::setprecision(4) << r2 << "\t";

    x = dr[0];
    y = dr[1];
    z = dr[2];

    rad = 0.0;
    //  std::cerr << std::endl << "\t";
    for (j = h_basis.stop_list[i]; j < h_basis.stop_list[i + 1]; j++) {
      //      std::cerr << std::setw(7) << std::setprecision(5) << zti << "\t";
      //      std::cerr << std::setw(7) << std::setprecision(5) << qc_shl_list[i].norm[j] << "\t";
      //      std::cerr << std::setw(7) << std::setprecision(5) << exp(-zti*r2) << "\t";
      //      std::cerr << "\n\t";

      rad = rad + exp(-h_basis.alpha[j] * r2) * h_basis.norm[j];
    }

    //    if (ishl%am == -1) {  // SP
    //      do ip = 1, ishl%nprim {
    //        zti = ishl%alpha(ip)
    //        rad(2) = rad(2) + exp(-zti*r2)*ishl%norm(2, ip)  // Px
    //      }
    //    }  // ---

    //    memset(ang, 0, 15*sizeof(double));
    std::fill(ang, ang + 15, 0);

    if (lspherical) {
      ic = h_basis.isgs[i];
      //      std::cerr << "isgs\t" << ic << std::endl;
      if (iam == 0) {
        h_basis.icgs[ic + 0] = rad;
      } else if (iam == -1) {
        h_basis.icgs[ic + 0] = rad;
        h_basis.icgs[ic + 1] = rad * x;
        h_basis.icgs[ic + 2] = rad * y;
        h_basis.icgs[ic + 3] = rad * z;
      } else if (iam == 1) {
        h_basis.icgs[ic + 0] = rad * x;
        h_basis.icgs[ic + 1] = rad * y;
        h_basis.icgs[ic + 2] = rad * z;
      } else if (iam == 2) {
        /*
        // l = 2, m = -2
        ang[0] = cd[0]*x*y;  // dxy
        // l = 2, m = -1
        ang[1] = cd[0]*y*z;  // dyz
        // l = 2, m = 0
        ang[2] = 0.5*(2.0*z*z - x*x - y*y);  // dxx, dyy, dzz
        ang[3] = -cd[0]*x*z;  // dxz
*/

        ang[0] = 1.732050807568877 * x * y;            // dxy
        ang[1] = 1.732050807568877 * y * z;            // dyz
        ang[2] = 0.5 * (2.0 * z * z - x * x - y * y);  // dxx, dyy, dzz
        ang[3] = -1.732050807568877 * x * z;           // dxz
        ang[4] = 0.86602540378443 * (x * x - y * y);   // dxx, dyy

        // --(test) --
        // ang(1) = x*y  // dxy
        // ang(2) = y*z  // dyz
        // ang(3) = 0.5_dp*(2.0_dp*z*z - x*x - y*y)  // dxx, dyy, dzz
        // ang(4) = -x*z  // dxz
        // ang(5) =  cd(2)*(x*x - y*y)  // dxx, dyy
        // --
        h_basis.icgs[ic + 0] = rad * ang[0];
        h_basis.icgs[ic + 1] = rad * ang[1];
        h_basis.icgs[ic + 2] = rad * ang[2];
        h_basis.icgs[ic + 3] = rad * ang[3];
        h_basis.icgs[ic + 4] = rad * ang[4];
      } else if (iam == 3) {
        ang[0] = y * (cf[1] * x * x - cf[0] * y * y);  // xxy, yyy
        ang[1] = cf[2] * x * y * z;
        ang[2] = y * (cf[4] * z * z - cf[3] * (x * x + y * y));
        ang[3] = z * (z * z - cf[5] * (x * x + y * y));
        ang[4] = -x * (cf[4] * z * z - cf[3] * (x * x + y * y));
        ang[5] = z * cf[6] * (x * x - y * y);
        ang[6] = x * (cf[1] * y * y - cf[0] * x * x);

        h_basis.icgs[ic + 0] = rad * ang[0];
        h_basis.icgs[ic + 1] = rad * ang[1];
        h_basis.icgs[ic + 2] = rad * ang[2];
        h_basis.icgs[ic + 3] = rad * ang[3];
        h_basis.icgs[ic + 4] = rad * ang[4];
        h_basis.icgs[ic + 5] = rad * ang[5];
        h_basis.icgs[ic + 6] = rad * ang[6];
      } else if (iam == 4) {
        // m = -4
        ang[0] = cg[0] * (x * x * x * y - x * y * y * y);  // xxxy, xyyy
        // m = -3
        ang[1] = y * z * (cg[1] * x * x - cg[2] * y * y);  // (2, 1, 1) (0, 3, 1)
        // m = -2
        ang[2] = x * y * cg[3] * (-x * x - y * y) + cg[4] * x * y * z * z;
        // m = -1
        ang[3] = -cg[5] * x * x * y * z - cg[5] * y * y * y * z + cg[6] * y * z * z * z;
        // m = 0
        ang[4] = 0.375 * (x * x * x * x + y * y * y * y + 2.0 * x * x * y * y) + z * z * z * z - 3.0 * z * z * (x * x + y * y);
        // m = 1
        ang[5] = cg[5] * x * x * x * z + cg[5] * x * y * y * z - cg[6] * x * z * z * z;
        // m = 2
        ang[6] = cg[7] * (y * y * y * y - x * x * x * x) + cg[8] * z * z * (x * x - y * y);
        // m = 3
        ang[7] = x * z * (cg[1] * y * y - cg[2] * x * x);  // (1, 2, 1) (3, 0, 1)
        // m = 4
        ang[8] = cg[9] * (x * x * x * x + y * y * y * y) - cg[10] * x * x * y * y;

        h_basis.icgs[ic + 0] = rad * ang[0];
        h_basis.icgs[ic + 1] = rad * ang[1];
        h_basis.icgs[ic + 2] = rad * ang[2];
        h_basis.icgs[ic + 3] = rad * ang[3];
        h_basis.icgs[ic + 4] = rad * ang[4];
        h_basis.icgs[ic + 5] = rad * ang[5];
        h_basis.icgs[ic + 6] = rad * ang[6];
        h_basis.icgs[ic + 7] = rad * ang[7];
        h_basis.icgs[ic + 8] = rad * ang[8];
      }
    }
    /*
    else {
      // cartesian GTO
      // print *, 'ish am ic ', is, ishl%am, ishl%icgs

      ic = ishl%icgs - 1
      if (ishl%am == 0) {
        icgs(ic+1) = rad(1)
      } else if (ishl%am == -1) {
        icgs(ic+1) = rad(1)
        icgs(ic+2) = rad(2)*x
        icgs(ic+3) = rad(2)*y
        icgs(ic+4) = rad(2)*z
      } else if (ishl%am == 1) {
        icgs(ic+1) = rad(1)*x
        icgs(ic+2) = rad(1)*y
        icgs(ic+3) = rad(1)*z
      } else if (ishl%am == 2) {
        ang(1) = x*x  // dxx
        ang(2) = x*y  // cd(1)*x*y  // dxy
        ang(3) = x*z  // cd(1)*x*z  // dxz
        ang(4) = y*y  // dyy
        ang(5) = y*z  // cd(1)*y*z  // dyz
        ang(6) = z*z  // dzz

        icgs(ic+1) = rad(1)*ang(1)
        icgs(ic+2) = rad(1)*ang(2)
        icgs(ic+3) = rad(1)*ang(3)
        icgs(ic+4) = rad(1)*ang(4)
        icgs(ic+5) = rad(1)*ang(5)
        icgs(ic+6) = rad(1)*ang(6)
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

        icgs(ic+1) = rad(1)*ang(1)
        icgs(ic+2) = rad(1)*ang(2)
        icgs(ic+3) = rad(1)*ang(3)
        icgs(ic+4) = rad(1)*ang(4)
        icgs(ic+5) = rad(1)*ang(5)
        icgs(ic+6) = rad(1)*ang(6)
        icgs(ic+7) = rad(1)*ang(7)
        icgs(ic+8) = rad(1)*ang(8)
        icgs(ic+9) = rad(1)*ang(9)
        icgs(ic+10) = rad(1)*ang(10)
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

        icgs(ic+1) = rad(1)*ang(1)
        icgs(ic+2) = rad(1)*ang(2)
        icgs(ic+3) = rad(1)*ang(3)
        icgs(ic+4) = rad(1)*ang(4)
        icgs(ic+5) = rad(1)*ang(5)
        icgs(ic+6) = rad(1)*ang(6)
        icgs(ic+7) = rad(1)*ang(7)
        icgs(ic+8) = rad(1)*ang(8)
        icgs(ic+9) = rad(1)*ang(9)
        icgs(ic+10) = rad(1)*ang(10)
        icgs(ic+11) = rad(1)*ang(11)
        icgs(ic+12) = rad(1)*ang(12)
        icgs(ic+13) = rad(1)*ang(13)
        icgs(ic+14) = rad(1)*ang(14)
        icgs(ic+15) = rad(1)*ang(15)
      }
    }
    */
    // std::cerr << std::endl;
  }
}

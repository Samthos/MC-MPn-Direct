#include "atomic_orbital.h"

#include <cmath>
#include <iostream>

#include "cartesian_poly.h"

#ifdef HAVE_CUDA
#define HOSTDEVICE __host__ __device__
#else 
#define HOSTDEVICE
#endif

Atomic_Orbital::Atomic_Orbital(const int contraction_begin_in,
    const int contraction_end_in,
    const int contraction_index_in,
    const int ao_index_in,
    const int angular_momentum_in,
    const bool is_spherical_in,
    const Point& pos_in) : 
  contraction_begin(contraction_begin_in),
  contraction_end(contraction_end_in),
  contraction_index(contraction_index_in),
  ao_index(ao_index_in),
  angular_momentum(angular_momentum_in),
  is_spherical(is_spherical_in),
  pos(pos_in)
{
}

// 
// Functions to evaluation contractions
//
HOSTDEVICE
void Atomic_Orbital::evaluate_contraction(double* contraction_amplitudes,
    double* contraction_exponent,
    double* contraction_coeficient,
    const Point& walker_pos) {
  double r2 = Point::distance_squared(walker_pos, pos);
  contraction_amplitudes[contraction_index] = 0.0;
  for (auto i = contraction_begin; i < contraction_end; i++) {
    contraction_amplitudes[contraction_index] += exp(-contraction_exponent[i] * r2) * contraction_coeficient[i];
  }
}

HOSTDEVICE
void Atomic_Orbital::evaluate_contraction_with_derivative(
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    double* contraction_exponent,
    double* contraction_coeficient,
    const Point& walker_pos) {
  double r2 = Point::distance_squared(walker_pos, pos);
  contraction_amplitudes[contraction_index] = 0.0;
  contraction_amplitudes_derivative[contraction_index] = 0.0;
  for (auto i = contraction_begin; i < contraction_end; i++) {
    double alpha = contraction_exponent[i];
    double exponential = exp(-alpha * r2) * contraction_coeficient[i];
    contraction_amplitudes[contraction_index] += exponential;
    contraction_amplitudes_derivative[contraction_index] -= 2.0 * alpha * exponential;
  }
}

HOSTDEVICE
double Atomic_Orbital::calculate_r2(const double walker_pos[3]) {
  double r2 = 0.0;
  r2 += (pos[0] - walker_pos[0]) * (pos[0] - walker_pos[0]);
  r2 += (pos[1] - walker_pos[1]) * (pos[1] - walker_pos[1]);
  r2 += (pos[2] - walker_pos[2]) * (pos[2] - walker_pos[2]);
  return r2;
}

HOSTDEVICE
void Atomic_Orbital::evaluate_ao(double* ao_amplitudes, double* contraction_amplitudes, const Point& walker_pos) {
  ao_amplitudes += ao_index;
  double x = walker_pos[0] - pos[0];
  double y = walker_pos[1] - pos[1];
  double z = walker_pos[2] - pos[2];

  if (is_spherical) {
    switch (angular_momentum) {
      case 0:           evaluate_s(ao_amplitudes, contraction_amplitudes[contraction_index], x, y ,z); break;
      case 1:           evaluate_p(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 2: evaluate_spherical_d(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 3: evaluate_spherical_f(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 4: evaluate_spherical_g(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
    }
  } else {
    switch (angular_momentum) {
      case 0:           evaluate_s(ao_amplitudes, contraction_amplitudes[contraction_index], x, y ,z); break;
      case 1:           evaluate_p(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 2: evaluate_cartesian_d(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 3: evaluate_cartesian_f(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
      case 4: evaluate_cartesian_g(ao_amplitudes, contraction_amplitudes[contraction_index], x, y, z); break;
    }
  }
}
HOSTDEVICE
void Atomic_Orbital::evaluate_ao_dx(double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, const Point& walker_pos) {
  ao_amplitudes += ao_index;
  double x = walker_pos[0] - pos[0];
  double y = walker_pos[1] - pos[1];
  double z = walker_pos[2] - pos[2];

  if (is_spherical) {
    switch (angular_momentum) {
      case 0:           evaluate_s_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_spherical_d_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_spherical_f_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_spherical_g_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  } else {
    switch (angular_momentum) {
      case 0:           evaluate_s_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_cartesian_d_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_cartesian_f_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_cartesian_g_dx(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  }
}
HOSTDEVICE
void Atomic_Orbital::evaluate_ao_dy(double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, const Point& walker_pos) {
  ao_amplitudes += ao_index;
  double x = walker_pos[0] - pos[0];
  double y = walker_pos[1] - pos[1];
  double z = walker_pos[2] - pos[2];

  if (is_spherical) {
    switch (angular_momentum) {
      case 0:           evaluate_s_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_spherical_d_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_spherical_f_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_spherical_g_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  } else {
    switch (angular_momentum) {
      case 0:           evaluate_s_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_cartesian_d_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_cartesian_f_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_cartesian_g_dy(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  }
}
HOSTDEVICE
void Atomic_Orbital::evaluate_ao_dz(double* ao_amplitudes, double* contraction_amplitudes, double* contraction_amplitude_derivatives, const Point& walker_pos) {
  ao_amplitudes += ao_index;
  double x = walker_pos[0] - pos[0];
  double y = walker_pos[1] - pos[1];
  double z = walker_pos[2] - pos[2];

  if (is_spherical) {
    switch (angular_momentum) {
      case 0:           evaluate_s_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_spherical_d_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_spherical_f_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_spherical_g_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  } else {
    switch (angular_momentum) {
      case 0:           evaluate_s_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y ,z); break;
      case 1:           evaluate_p_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 2: evaluate_cartesian_d_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 3: evaluate_cartesian_f_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
      case 4: evaluate_cartesian_g_dz(ao_amplitudes, contraction_amplitudes[contraction_index], contraction_amplitude_derivatives[contraction_index], x, y, z); break;
    }
  }
}

//
// Function to evaluate shell that are the same for spherical and cartesian basis
//
HOSTDEVICE
void Atomic_Orbital::evaluate_s(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  ao_amplitudes[0] = rad;
}
HOSTDEVICE
void Atomic_Orbital::evaluate_p(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  ao_amplitudes[X] = rad * x;
  ao_amplitudes[Y] = rad * y;
  ao_amplitudes[Z] = rad * z;
}
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  using namespace Cartesian_Poly;
  evaluate_p(ao_amplitudes, rad, x, y, z);
  ao_amplitudes[ZZ] = z * ao_amplitudes[Z];
  ao_amplitudes[YZ] = y * ao_amplitudes[Z];
  ao_amplitudes[YY] = y * ao_amplitudes[Y];
  ao_amplitudes[XZ] = x * ao_amplitudes[Z];
  ao_amplitudes[XY] = x * ao_amplitudes[Y];
  ao_amplitudes[XX] = x * ao_amplitudes[X];
}

//
// Function to evaluate shell for cartesian basis
//
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_h(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
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


//
// Function to evaluate shell for spherical basis
//
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_d_shell(double* ao_amplitudes, double* ang) {
  using namespace Cartesian_Poly;
  constexpr double cd[] = {1.732050807568877, // sqrt(3)
                           0.86602540378443}; // 0.5 * sqrt(3)
  ao_amplitudes[0] =  cd[0] * ang[XY];
  ao_amplitudes[1] =  cd[0] * ang[YZ];
  ao_amplitudes[2] =  0.5 * (2.0 * ang[ZZ] - ang[XX] - ang[YY]);
  ao_amplitudes[3] = -cd[0] * ang[XZ];
  ao_amplitudes[4] =  cd[1] * (ang[XX] - ang[YY]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_f_shell(double* ao_amplitudes, double* ang) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_g_shell(double* ao_amplitudes, double* ang) {
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


//
// Function to orchestrate the evaluatation shells for spherical basis
//
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_d(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[6];
  evaluate_cartesian_d(&ang[0], rad, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_f(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[10];
  evaluate_cartesian_f(&ang[0], rad, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_g(double *ao_amplitudes, const double& rad, const double& x, const double& y, const double& z) {
  double ang[15];
  evaluate_cartesian_g(&ang[0], rad, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}


//
// Function to orchestrate the evaluatation dx atomic orbital derivatives
//
HOSTDEVICE
void Atomic_Orbital::evaluate_s_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = x * rad_derivative; 
}
HOSTDEVICE
void Atomic_Orbital::evaluate_p_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[X] = x * ucgs[X] + dcgs;
  ao_amplitudes[Y] = x * ucgs[Y];
  ao_amplitudes[Z] = x * ucgs[Z];
}
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_d_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_f_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_g_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_d_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_f_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_g_dx(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dx(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}

//
// Function to orchestrate the evaluatation dy atomic orbital derivatives
//
HOSTDEVICE
void Atomic_Orbital::evaluate_s_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = y * rad_derivative;
}
HOSTDEVICE
void Atomic_Orbital::evaluate_p_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[0] = y * ucgs[X];
  ao_amplitudes[1] = y * ucgs[Y] + dcgs;
  ao_amplitudes[2] = y * ucgs[Z];
}
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_d_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_f_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_g_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_d_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_f_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_g_dy(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dy(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}

//
// Function to orchestrate the evaluatation dz atomic orbital derivatives
//
HOSTDEVICE
void Atomic_Orbital::evaluate_s_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
	ao_amplitudes[0] = z * rad_derivative;
}
HOSTDEVICE
void Atomic_Orbital::evaluate_p_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  using namespace Cartesian_Poly;

  double dcgs;
  double ucgs[3];
  evaluate_s(&dcgs, rad, x, y, z);
  evaluate_p(&ucgs[0], rad_derivative, x, y, z);

  ao_amplitudes[X] = z * ucgs[X];
  ao_amplitudes[Y] = z * ucgs[Y];
  ao_amplitudes[Z] = z * ucgs[Z] + dcgs;
}
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_d_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_f_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_cartesian_g_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
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
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_d_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[6];
  evaluate_cartesian_d_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_d_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_f_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[10];
  evaluate_cartesian_f_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_f_shell(ao_amplitudes, &ang[0]);
}
HOSTDEVICE
void Atomic_Orbital::evaluate_spherical_g_dz(double* ao_amplitudes, const double &rad, const double &rad_derivative, const double &x, const double &y, const double &z) {
  double ang[15];
  evaluate_cartesian_g_dz(&ang[0], rad, rad_derivative, x, y, z);
  evaluate_spherical_g_shell(ao_amplitudes, &ang[0]);
}
#undef HOSTDEVICE

//
// Created by aedoran on 12/18/19.
//

#include <cmath>

#include "correlation_factor_function.h"
#include "correlation_factor.h"

#ifdef HAVE_CUDA
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

HOSTDEVICE
Correlation_Factor_Function* create_correlation_factor_function(CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta) {
  Correlation_Factor_Function* correlation_factor = nullptr;
  switch (correlation_factor_id) {
    case CORRELATION_FACTORS::Linear: correlation_factor =  new Linear_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Rational: correlation_factor =  new Rational_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Slater: correlation_factor =  new Slater_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Slater_Linear: correlation_factor =  new Slater_Linear_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Gaussian: correlation_factor =  new Gaussian_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Cusped_Gaussian: correlation_factor =  new Cusped_Gaussian_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Yukawa_Coulomb: correlation_factor =  new Yukawa_Coulomb_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Jastrow: correlation_factor =  new Jastrow_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::ERFC: correlation_factor =  new ERFC_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::ERFC_Linear: correlation_factor =  new ERFC_Linear_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Tanh: correlation_factor =  new Tanh_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::ArcTan: correlation_factor =  new ArcTan_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Logarithm: correlation_factor =  new Logarithm_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Hybrid: correlation_factor =  new Hybrid_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Two_Parameter_Rational: correlation_factor =  new Two_Parameter_Rational_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Higher_Rational: correlation_factor =  new Higher_Rational_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Cubic_Slater: correlation_factor =  new Cubic_Slater_Correlation_Factor_Function(gamma, beta); break;
    case CORRELATION_FACTORS::Higher_Jastrow: correlation_factor =  new Higher_Jastrow_Correlation_Factor_Function(gamma, beta); break;
  }
  return correlation_factor;
}

HOSTDEVICE
Correlation_Factor_Function::Correlation_Factor_Function(double gamma_, double beta_) :
    gamma(gamma_), 
    beta(beta_) {
}

HOSTDEVICE
Correlation_Factor_Function::Correlation_Factor_Function(double gamma_, double beta_, double default_gamma, double default_beta) {
  gamma = gamma_;
  if (gamma_ < 0) {
    gamma = default_gamma;
  }
  beta = beta_;
  if (beta_ < 0) {
    beta = default_beta;
  }
}

HOSTDEVICE double Linear_Correlation_Factor_Function::f12(double r12) {
  return r12;
}
HOSTDEVICE double Linear_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0;
}
HOSTDEVICE double Linear_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double Linear_Correlation_Factor_Function::f12_c(double r12) { 
  return 0.0;
}
HOSTDEVICE double Linear_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Linear_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Rational_Correlation_Factor_Function::f12(double r12) {
  return gamma * r12 / (gamma + r12);
}
HOSTDEVICE double Rational_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  return -2.0 * d1 * d1 * d1;
}
HOSTDEVICE double Rational_Correlation_Factor_Function::f12_b(double r12) { 
  return 0.0; 
}
HOSTDEVICE double Rational_Correlation_Factor_Function::f12_c(double r12) {
  return gamma / (gamma + r12);
}
HOSTDEVICE double Rational_Correlation_Factor_Function::f12_d(double r12) {
  double d1 = gamma + r12;
  return -gamma / (d1 * d1);
}
bool Rational_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Slater_Correlation_Factor_Function::f12(double r12) {
  // return (1.0-exp(-gamma*r12))/gamma;
  return -exp(-gamma * r12) / gamma;
}
HOSTDEVICE double Slater_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
HOSTDEVICE double Slater_Correlation_Factor_Function::f12_b(double r12) {
  return gamma * exp(-gamma * r12);
}
HOSTDEVICE double Slater_Correlation_Factor_Function::f12_c(double r12) {
  return exp(-gamma * r12);
}
HOSTDEVICE double Slater_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Slater_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Slater_Linear_Correlation_Factor_Function::f12(double r12) {
  return r12 * exp(-gamma * r12);
}
HOSTDEVICE double Slater_Linear_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
HOSTDEVICE double Slater_Linear_Correlation_Factor_Function::f12_b(double r12) {
  return gamma * (4.0 - r12 * gamma) * exp(-gamma * r12);
}
HOSTDEVICE double Slater_Linear_Correlation_Factor_Function::f12_c(double r12) {
  return exp(-gamma * r12);
}
HOSTDEVICE double Slater_Linear_Correlation_Factor_Function::f12_d(double r12) {
  return -gamma * exp(-gamma * r12);
}
bool Slater_Linear_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Gaussian_Correlation_Factor_Function::f12(double r12) {
  return 1.0-exp(-gamma*r12*r12);
  // return -exp(-gamma * r12 * r12);
}
HOSTDEVICE double Gaussian_Correlation_Factor_Function::f12_a(double r12) { 
  return 0.0;
}
HOSTDEVICE double Gaussian_Correlation_Factor_Function::f12_b(double r12) {
  double r12_2 = r12 * r12;
  double er12_2 = exp(-gamma * r12_2);
  return 2.0 * gamma * er12_2 * (2 * gamma * r12_2 - 3);
}
HOSTDEVICE double Gaussian_Correlation_Factor_Function::f12_c(double r12) {
  return 0.0; 
}
HOSTDEVICE double Gaussian_Correlation_Factor_Function::f12_d(double r12) {
  return 2.0 * gamma * exp(-gamma * r12 * r12);
}
bool Gaussian_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Cusped_Gaussian_Correlation_Factor_Function::f12(double r12) {
  if (r12 == 0.0) {
    return 0.0;
  }
  return (1.0 - exp(-gamma * r12 * r12)) / (gamma * r12);
}
HOSTDEVICE double Cusped_Gaussian_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * exp(-gamma * r12 * r12);
}
HOSTDEVICE double Cusped_Gaussian_Correlation_Factor_Function::f12_b(double r12) {
  return 4.0 * gamma * r12 * exp(-gamma * r12 * r12);
}
HOSTDEVICE double Cusped_Gaussian_Correlation_Factor_Function::f12_c(double r12) {
  if (r12 == 0.0) {
    return 1.0;
  }
  double er12_2 = exp(-gamma * r12 * r12);
  return (er12_2 - 1.0) / (gamma * r12 * r12) + 2.0 * er12_2;
}
HOSTDEVICE double Cusped_Gaussian_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Cusped_Gaussian_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Yukawa_Coulomb_Correlation_Factor_Function::f12(double r12) {
  if (r12 == 0.0) {
    return -2.0 / gamma;
  }
  return 2.0 * (exp(-gamma * r12) - 1.0) / (gamma * gamma * r12);
}
HOSTDEVICE double Yukawa_Coulomb_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
HOSTDEVICE double Yukawa_Coulomb_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double Yukawa_Coulomb_Correlation_Factor_Function::f12_c(double r12) {
  if (r12 == 0.0) {
    return 1.0;
  }
  return 2.0 * (1.0 - (1.0 + gamma * r12) * exp(-gamma * r12)) / (gamma * gamma * r12 * r12);
}
HOSTDEVICE double Yukawa_Coulomb_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Yukawa_Coulomb_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Jastrow_Correlation_Factor_Function::f12(double r12) {
  return exp(r12 / (1 + gamma * r12));
}
HOSTDEVICE double Jastrow_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = (1 + gamma * r12);
  return -1.0 * exp(r12 / (1 + gamma * r12)) * (2.0 + r12 + 2.0 * gamma * r12) / (d1 * d1 * d1 * d1);
}
HOSTDEVICE double Jastrow_Correlation_Factor_Function::f12_b(double r12) { 
  return 0.0;
}
HOSTDEVICE double Jastrow_Correlation_Factor_Function::f12_c(double r12) {
  return exp(r12 / (1 + gamma * r12)) / (1 + gamma * r12);
}
HOSTDEVICE double Jastrow_Correlation_Factor_Function::f12_d(double r12) {
  return -exp(r12 / (1 + gamma * r12)) * gamma / ((1 + gamma * r12) * (1 + gamma * r12));
}
bool Jastrow_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double ERFC_Correlation_Factor_Function::f12(double r12) {
  return -sqrt_pi * (erfc(gamma * r12)) / (2 * gamma);
}
HOSTDEVICE double ERFC_Correlation_Factor_Function::f12_a(double r12) {
  double er12 = gamma * gamma * r12 * r12;
  return 2.0 * exp(-er12) * (er12 - 1);
}
HOSTDEVICE double ERFC_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double ERFC_Correlation_Factor_Function::f12_c(double r12) {
  return exp(-gamma * gamma * r12 * r12);
}
HOSTDEVICE double ERFC_Correlation_Factor_Function::f12_d(double r12) { 
  return 0.0;
}
bool ERFC_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double ERFC_Linear_Correlation_Factor_Function::f12(double r12) {
  return r12 * erfc(gamma * r12);
}
HOSTDEVICE double ERFC_Linear_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * erfc(gamma * r12);
}
HOSTDEVICE double ERFC_Linear_Correlation_Factor_Function::f12_b(double r12) {
  auto d = gamma * gamma * r12 * r12;
  return exp(-d) * (8.0 * gamma - 4.0 * gamma * d) / sqrt_pi;
}
HOSTDEVICE double ERFC_Linear_Correlation_Factor_Function::f12_c(double r12) {
  return erfc(gamma * r12);
}
HOSTDEVICE double ERFC_Linear_Correlation_Factor_Function::f12_d(double r12) {
  return -2.0 * exp(-gamma * gamma * r12 * r12) * gamma / sqrt_pi;
}
bool ERFC_Linear_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Tanh_Correlation_Factor_Function::f12(double r12) {
  return tanh(gamma * r12) / gamma;
}
HOSTDEVICE double Tanh_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 / (cosh(gamma * r12) * cosh(gamma * r12));
}
HOSTDEVICE double Tanh_Correlation_Factor_Function::f12_b(double r12) {
  return 2.0 * gamma * tanh(gamma * r12) / (cosh(gamma * r12) * cosh(gamma * r12));
}
HOSTDEVICE double Tanh_Correlation_Factor_Function::f12_c(double r12) {
  return 1 / (cosh(gamma * r12) * cosh(gamma * r12));
}
HOSTDEVICE double Tanh_Correlation_Factor_Function::f12_d(double r12) { 
  return 0.0; 
}
bool Tanh_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double ArcTan_Correlation_Factor_Function::f12(double r12) {
  return atan(gamma * r12) / gamma;
}
HOSTDEVICE double ArcTan_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = 1.0 + gamma * gamma * r12 * r12;
  return -2.0 / (d1 * d1);
}
HOSTDEVICE double ArcTan_Correlation_Factor_Function::f12_b(double r12) { 
  return 0.0;
}
HOSTDEVICE double ArcTan_Correlation_Factor_Function::f12_c(double r12) {
  return 1.0 / (1.0 + gamma * gamma * r12 * r12);
}
HOSTDEVICE double ArcTan_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0; 
}
bool ArcTan_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Logarithm_Correlation_Factor_Function::f12(double r12) {
  return log(1.0 + gamma * r12) / gamma;
}
HOSTDEVICE double Logarithm_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = 1.0 + gamma * r12;
  return -2.0 / (d1 * d1);
}
HOSTDEVICE double Logarithm_Correlation_Factor_Function::f12_b(double r12) {
  double d1 = 1.0 + gamma * r12;
  return -gamma / (d1 * d1);
}
HOSTDEVICE double Logarithm_Correlation_Factor_Function::f12_c(double r12) {
  return 1.0 / (1.0 + gamma * r12);
}
HOSTDEVICE double Logarithm_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Logarithm_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}


HOSTDEVICE double Hybrid_Correlation_Factor_Function::f12(double r12) {
  return -0.5 * exp(-gamma * r12) / gamma + 0.5 * gamma * r12 / (gamma + r12);
}
HOSTDEVICE double Hybrid_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  return -exp(-gamma * r12) - d1 * d1 * d1;
}
HOSTDEVICE double Hybrid_Correlation_Factor_Function::f12_b(double r12) {
  return 0.5 * gamma * exp(-gamma * r12);
}
HOSTDEVICE double Hybrid_Correlation_Factor_Function::f12_c(double r12) {
  return 0.5 * (exp(-gamma * r12) + gamma / (gamma + r12));
}
HOSTDEVICE double Hybrid_Correlation_Factor_Function::f12_d(double r12) {
  return -0.5 * gamma / ((gamma + r12) * (gamma + r12));
}
bool Hybrid_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Two_Parameter_Rational_Correlation_Factor_Function::f12(double r12) {
  return gamma * r12 / (gamma + beta * r12);
}
HOSTDEVICE double Two_Parameter_Rational_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = gamma / (gamma + beta * r12);
  return -2.0 * d1 * d1 * d1;
}
HOSTDEVICE double Two_Parameter_Rational_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double Two_Parameter_Rational_Correlation_Factor_Function::f12_c(double r12) {
  return gamma / (gamma + beta * r12);
}
HOSTDEVICE double Two_Parameter_Rational_Correlation_Factor_Function::f12_d(double r12) {
  double d1 = gamma + beta * r12;
  return -beta * gamma / (d1 * d1);
}
bool Two_Parameter_Rational_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Higher_Rational_Correlation_Factor_Function::f12(double r12) {
  double denominator = beta + r12;
  return gamma * r12 / (2.0 * (gamma + r12)) + beta * beta * r12 / (2.0 * (denominator * denominator));
}
HOSTDEVICE double Higher_Rational_Correlation_Factor_Function::f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  double d2 = beta / (beta + r12);
  return -d1 * d1 * d1 + (2 * r12 - beta) * (d2 * d2 * d2 * d2) / beta;
}
HOSTDEVICE double Higher_Rational_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double Higher_Rational_Correlation_Factor_Function::f12_c(double r12) {
  return 0.5 *
         (gamma / (gamma + r12) + beta * beta / ((beta + r12) * (beta + r12)));
}
HOSTDEVICE double Higher_Rational_Correlation_Factor_Function::f12_d(double r12) {
  double d1 = gamma / (gamma + r12);
  double d2 = beta / (beta + r12);
  return -0.5 * d1 * d1 / gamma - d2 * d2 * d2 / beta;
}
bool Higher_Rational_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Cubic_Slater_Correlation_Factor_Function::f12(double r12) {
  // return (1.0-exp(-gamma*r12))/gamma+exp(-beta*r12*r12*r12)-1;
  return (-exp(-gamma * r12)) / gamma + exp(-beta * r12 * r12 * r12);
}
HOSTDEVICE double Cubic_Slater_Correlation_Factor_Function::f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
HOSTDEVICE double Cubic_Slater_Correlation_Factor_Function::f12_b(double r12) {
  double er12 = exp(-gamma * r12);
  double r12_3 = r12 * r12 * r12;
  double er12_3 = exp(-beta * r12_3);
  return gamma * er12 + r12 * er12_3 * beta * (12.0 - 9.0 * r12_3 * beta);
}
HOSTDEVICE double Cubic_Slater_Correlation_Factor_Function::f12_c(double r12) {
  return exp(-gamma * r12);
}
HOSTDEVICE double Cubic_Slater_Correlation_Factor_Function::f12_d(double r12) {
  double r12_3 = r12 * r12 * r12;
  return -3.0 * beta * r12 * exp(-beta * r12_3);
}
bool Cubic_Slater_Correlation_Factor_Function::f12_d_is_zero() {
  return false;
}


HOSTDEVICE double Higher_Jastrow_Correlation_Factor_Function::f12(double r12) {
  double fac1 = (1 - exp(-beta * r12)) / beta;
  return exp(fac1 / (1 + gamma * fac1));
}
HOSTDEVICE double Higher_Jastrow_Correlation_Factor_Function::f12_a(double r12) {
  double er12 = exp(beta * r12);
  double fac1 = beta + gamma;
  double fac2 = beta * beta * exp(1 / fac1 + beta * (r12 + 1 / (gamma * fac1 - fac1 * fac1 * er12)));
  double fac3 = gamma - fac1 * er12;
  return fac2 * (-(2.0 + beta * r12) * gamma * gamma +
           exp(2 * beta * r12) * (beta * r12 - 2.0) * fac1 * fac1 +
           er12 * (4.0 * gamma * (fac1)-beta * beta * r12)) /
          (fac3 * fac3 * fac3 * fac3);
}
HOSTDEVICE double Higher_Jastrow_Correlation_Factor_Function::f12_b(double r12) {
  return 0.0;
}
HOSTDEVICE double Higher_Jastrow_Correlation_Factor_Function::f12_c(double r12) {
  double er12 = exp(beta * r12);
  double fac1 = beta + gamma;
  double fac2 = beta * beta * exp(1 / fac1 + beta * (r12 + 1 / (gamma * fac1 - fac1 * fac1 * er12)));
  double fac3 = gamma - fac1 * er12;
  return fac2 / (fac3 * fac3);
}
HOSTDEVICE double Higher_Jastrow_Correlation_Factor_Function::f12_d(double r12) {
  return 0.0;
}
bool Higher_Jastrow_Correlation_Factor_Function::f12_d_is_zero() {
  return true;
}

#undef HOSTDEVICE

//
// Created by aedoran on 12/18/19.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "correlation_factors.h"

Correlation_Factor* create_correlation_factor(const IOPs& iops) {
  Correlation_Factor* correlation_factor = nullptr;
  switch (iops.iopns[KEYS::F12_CORRELATION_FACTOR]) {
    case CORRELATION_FACTORS::Linear: correlation_factor =  new Linear_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Rational: correlation_factor =  new Rational_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Slater: correlation_factor =  new Slater_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Slater_Linear: correlation_factor =  new Slater_Linear_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Gaussian: correlation_factor =  new Gaussian_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Cusped_Gaussian: correlation_factor =  new Cusped_Gaussian_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Yukawa_Coulomb: correlation_factor =  new Yukawa_Coulomb_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Jastrow: correlation_factor =  new Jastrow_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::ERFC: correlation_factor =  new ERFC_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::ERFC_Linear: correlation_factor =  new ERFC_Linear_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Tanh: correlation_factor =  new Tanh_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::ArcTan: correlation_factor =  new ArcTan_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Logarithm: correlation_factor =  new Logarithm_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Hybrid: correlation_factor =  new Hybrid_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Two_Parameter_Rational: correlation_factor =  new Two_Parameter_Rational_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Higher_Rational: correlation_factor =  new Higher_Rational_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Cubic_Slater: correlation_factor =  new Cubic_Slater_Correlation_Factor(iops); break;
    case CORRELATION_FACTORS::Higher_Jastrow: correlation_factor =  new Higher_Jastrow_Correlation_Factor(iops); break;
  }
  return correlation_factor;
}
CORRELATION_FACTORS::CORRELATION_FACTORS string_to_correlation_factors(const std::string& str) {
  CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor;
  if (str == "Linear") {
    correlation_factor = CORRELATION_FACTORS::Linear;
  } else if (str == "Rational") {
    correlation_factor = CORRELATION_FACTORS::Rational;
  } else if (str == "Slater") {
    correlation_factor = CORRELATION_FACTORS::Slater;
  } else if (str == "Slater_Linear") {
    correlation_factor = CORRELATION_FACTORS::Slater_Linear;
  } else if (str == "Gaussian") {
    correlation_factor = CORRELATION_FACTORS::Gaussian;
  } else if (str == "Cusped_Gaussian") {
    correlation_factor = CORRELATION_FACTORS::Cusped_Gaussian;
  } else if (str == "Yukawa_Coulomb") {
    correlation_factor = CORRELATION_FACTORS::Yukawa_Coulomb;
  } else if (str == "Jastrow") {
    correlation_factor = CORRELATION_FACTORS::Jastrow;
  } else if (str == "ERFC") {
    correlation_factor = CORRELATION_FACTORS::ERFC;
  } else if (str == "ERFC_Linear") {
    correlation_factor = CORRELATION_FACTORS::ERFC_Linear;
  } else if (str == "Tanh") {
    correlation_factor = CORRELATION_FACTORS::Tanh;
  } else if (str == "ArcTan") {
    correlation_factor = CORRELATION_FACTORS::ArcTan;
  } else if (str == "Logarithm") {
    correlation_factor = CORRELATION_FACTORS::Logarithm;
  } else if (str == "Hybrid") {
    correlation_factor = CORRELATION_FACTORS::Hybrid;
  } else if (str == "Two_Parameter_Rational") {
    correlation_factor = CORRELATION_FACTORS::Two_Parameter_Rational;
  } else if (str == "Higher_Rational") {
    correlation_factor = CORRELATION_FACTORS::Higher_Rational;
  } else if (str == "Cubic_Slater") {
    correlation_factor = CORRELATION_FACTORS::Cubic_Slater;
  } else if (str == "Higher_Jastrow") {
    correlation_factor = CORRELATION_FACTORS::Higher_Jastrow;
  } else {
    std::cerr << "Correlation factor " << str << " not supported\n";
    exit(0);
  }
  return correlation_factor;
}
std::string correlation_factors_to_string(const CORRELATION_FACTORS::CORRELATION_FACTORS& correlation_factor) {
  std::string str;
  switch (correlation_factor) {
    case CORRELATION_FACTORS::Linear: str = "Linear"; break;
    case CORRELATION_FACTORS::Rational: str = "Rational"; break;
    case CORRELATION_FACTORS::Slater: str = "Slater"; break;
    case CORRELATION_FACTORS::Slater_Linear: str = "Slater_Linear"; break;
    case CORRELATION_FACTORS::Gaussian: str = "Gaussian"; break;
    case CORRELATION_FACTORS::Cusped_Gaussian: str = "Cusped_Gaussian"; break;
    case CORRELATION_FACTORS::Yukawa_Coulomb: str = "Yukawa_Coulomb"; break;
    case CORRELATION_FACTORS::Jastrow: str = "Jastrow"; break;
    case CORRELATION_FACTORS::ERFC: str = "ERFC"; break;
    case CORRELATION_FACTORS::ERFC_Linear: str = "ERFC_Linear"; break;
    case CORRELATION_FACTORS::Tanh: str = "Tanh"; break;
    case CORRELATION_FACTORS::ArcTan: str = "ArcTan"; break;
    case CORRELATION_FACTORS::Logarithm: str = "Logarithm"; break;
    case CORRELATION_FACTORS::Hybrid: str = "Hybrid"; break;
    case CORRELATION_FACTORS::Two_Parameter_Rational: str = "Two_Parameter_Rational"; break;
    case CORRELATION_FACTORS::Higher_Rational: str = "Higher_Rational"; break;
    case CORRELATION_FACTORS::Cubic_Slater: str = "Cubic_Slater"; break;
    case CORRELATION_FACTORS::Higher_Jastrow: str = "Higher_Jastrow"; break;
  }
  return str;
}

Correlation_Factor::Correlation_Factor(const IOPs& iops, double gamma_, double beta_) :
    f12p(iops.iopns[KEYS::ELECTRON_PAIRS]),
    f12p_a(iops.iopns[KEYS::ELECTRON_PAIRS]),
    f12p_c(iops.iopns[KEYS::ELECTRON_PAIRS]),
    f12o(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    f12o_b(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    f12o_d(iops.iopns[KEYS::ELECTRONS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    f13(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    f23(iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRONS], 0.0),
    gamma(gamma_), 
    beta(beta_) 
{
  if (iops.bopns[KEYS::F12_GAMMA]) {
    gamma = iops.dopns[KEYS::F12_GAMMA];
  }
  if (iops.bopns[KEYS::F12_BETA]) {
    beta = iops.dopns[KEYS::F12_BETA];
  }
}

double Correlation_Factor::distance(const std::array<double, 3>& p1, const std::array<double, 3>& p2) {
  std::array<double, 3> dr{};
  std::transform(p1.begin(), p1.end(), p2.begin(), dr.begin(), std::minus<>());
  return sqrt(std::inner_product(dr.begin(), dr.end(), dr.begin(),0.0));
}

void Correlation_Factor::update(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = calculate_f12(electron_pair_list->r12[ip]);
    f12p_a[ip] = calculate_f12_a(electron_pair_list->r12[ip]);
    f12p_c[ip] = calculate_f12_c(electron_pair_list->r12[ip]);
  }
  
  for(int io = 0; io < electron_list->size();io++) {
    for(int jo = 0; jo < electron_list->size();jo++) {
      if (jo != io) {
        auto dr = distance(electron_list->pos[io], electron_list->pos[jo]);
        f12o[io * electron_list->size() + jo]  = calculate_f12(dr);
        f12o_b[io * electron_list->size() + jo] =  calculate_f12_b(dr);
        f12o_d[io * electron_list->size() + jo] =  calculate_f12_d(dr);
      } else {
        f12o[io * electron_list->size() + jo]   = 0.0;
        f12o_b[io * electron_list->size() + jo] = 0.0;
        f12o_d[io * electron_list->size() + jo] = 0.0;
      }
    }
  }

  for(int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for(int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = calculate_f12(distance(electron_pair_list->pos1[ip], electron_list->pos[io]));
      f23[ip * electron_list->size() + io] = calculate_f12(distance(electron_pair_list->pos2[ip], electron_list->pos[io]));
    }
  }
}


double Linear_Correlation_Factor::calculate_f12(double r12) {
  return r12;
}
double Linear_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0;
}
double Linear_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double Linear_Correlation_Factor::calculate_f12_c(double r12) { 
  return 0.0;
}
double Linear_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Linear_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Rational_Correlation_Factor::calculate_f12(double r12) {
  return gamma * r12 / (gamma + r12);
}
double Rational_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  return -2.0 * d1 * d1 * d1;
}
double Rational_Correlation_Factor::calculate_f12_b(double r12) { 
  return 0.0; 
}
double Rational_Correlation_Factor::calculate_f12_c(double r12) {
  return gamma / (gamma + r12);
}
double Rational_Correlation_Factor::calculate_f12_d(double r12) {
  double d1 = gamma + r12;
  return -gamma / (d1 * d1);
}
bool Rational_Correlation_Factor::f12_d_is_zero() {
  return false;
}


void Slater_Correlation_Factor::update(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = calculate_f12(electron_pair_list->r12[ip]);
    f12p_a[ip] = 2.0 * gamma * f12p[ip];
    f12p_c[ip] = -gamma * f12p[ip];
  }
  
  for(int io = 0, idx = 0; io < electron_list->size(); io++) {
    for(int jo = 0; jo < electron_list->size(); jo++, idx++) {
      if (jo != io) {
        f12o[idx] = calculate_f12(distance(electron_list->pos[io], electron_list->pos[jo]));
        f12o_b[idx] =  -gamma * gamma * f12o[idx];
      } else {
        f12o[idx]   = 0.0;
        f12o_b[idx] = 0.0;
      }
    }
  }

  for(int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for(int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = calculate_f12(distance(electron_pair_list->pos1[ip], electron_list->pos[io]));
      f23[ip * electron_list->size() + io] = calculate_f12(distance(electron_pair_list->pos2[ip], electron_list->pos[io]));
    }
  }
}
double Slater_Correlation_Factor::calculate_f12(double r12) {
  // return (1.0-exp(-gamma*r12))/gamma;
  return -exp(-gamma * r12) / gamma;
}
double Slater_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
double Slater_Correlation_Factor::calculate_f12_b(double r12) {
  return gamma * exp(-gamma * r12);
}
double Slater_Correlation_Factor::calculate_f12_c(double r12) {
  return exp(-gamma * r12);
}
double Slater_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Slater_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Slater_Linear_Correlation_Factor::calculate_f12(double r12) {
  return r12 * exp(-gamma * r12);
}
double Slater_Linear_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
double Slater_Linear_Correlation_Factor::calculate_f12_b(double r12) {
  return gamma * (4.0 - r12 * gamma) * exp(-gamma * r12);
}
double Slater_Linear_Correlation_Factor::calculate_f12_c(double r12) {
  return exp(-gamma * r12);
}
double Slater_Linear_Correlation_Factor::calculate_f12_d(double r12) {
  return -gamma * exp(-gamma * r12);
}
bool Slater_Linear_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Gaussian_Correlation_Factor::calculate_f12(double r12) {
  return 1.0-exp(-gamma*r12*r12);
  // return -exp(-gamma * r12 * r12);
}
double Gaussian_Correlation_Factor::calculate_f12_a(double r12) { 
  return 0.0;
}
double Gaussian_Correlation_Factor::calculate_f12_b(double r12) {
  double r12_2 = r12 * r12;
  double er12_2 = exp(-gamma * r12_2);
  return 2.0 * gamma * er12_2 * (2 * gamma * r12_2 - 3);
}
double Gaussian_Correlation_Factor::calculate_f12_c(double r12) {
  return 0.0; 
}
double Gaussian_Correlation_Factor::calculate_f12_d(double r12) {
  return 2.0 * gamma * exp(-gamma * r12 * r12);
}
bool Gaussian_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Cusped_Gaussian_Correlation_Factor::calculate_f12(double r12) {
  if (r12 == 0.0) {
    return 0.0;
  }
  return (1.0 - exp(-gamma * r12 * r12)) / (gamma * r12);
}
double Cusped_Gaussian_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * exp(-gamma * r12 * r12);
}
double Cusped_Gaussian_Correlation_Factor::calculate_f12_b(double r12) {
  return 4.0 * gamma * r12 * exp(-gamma * r12 * r12);
}
double Cusped_Gaussian_Correlation_Factor::calculate_f12_c(double r12) {
  if (r12 == 0.0) {
    return 1.0;
  }
  double er12_2 = exp(-gamma * r12 * r12);
  return (er12_2 - 1.0) / (gamma * r12 * r12) + 2.0 * er12_2;
}
double Cusped_Gaussian_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Cusped_Gaussian_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Yukawa_Coulomb_Correlation_Factor::calculate_f12(double r12) {
  if (r12 == 0.0) {
    return -2.0 / gamma;
  }
  return 2.0 * (exp(-gamma * r12) - 1.0) / (gamma * gamma * r12);
}
double Yukawa_Coulomb_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
double Yukawa_Coulomb_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double Yukawa_Coulomb_Correlation_Factor::calculate_f12_c(double r12) {
  if (r12 == 0.0) {
    return 1.0;
  }
  return 2.0 * (1.0 - (1.0 + gamma * r12) * exp(-gamma * r12)) / (gamma * gamma * r12 * r12);
}
double Yukawa_Coulomb_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Yukawa_Coulomb_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Jastrow_Correlation_Factor::calculate_f12(double r12) {
  return exp(r12 / (1 + gamma * r12));
}
double Jastrow_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = (1 + gamma * r12);
  return -1.0 * exp(r12 / (1 + gamma * r12)) * (2.0 + r12 + 2.0 * gamma * r12) / (d1 * d1 * d1 * d1);
}
double Jastrow_Correlation_Factor::calculate_f12_b(double r12) { 
  return 0.0;
}
double Jastrow_Correlation_Factor::calculate_f12_c(double r12) {
  return exp(r12 / (1 + gamma * r12)) / (1 + gamma * r12);
}
double Jastrow_Correlation_Factor::calculate_f12_d(double r12) {
  return -exp(r12 / (1 + gamma * r12)) * gamma / ((1 + gamma * r12) * (1 + gamma * r12));
}
bool Jastrow_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double ERFC_Correlation_Factor::calculate_f12(double r12) {
  return -sqrt_pi * (erfc(gamma * r12)) / (2 * gamma);
}
double ERFC_Correlation_Factor::calculate_f12_a(double r12) {
  double er12 = gamma * gamma * r12 * r12;
  return 2.0 * exp(-er12) * (er12 - 1);
}
double ERFC_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double ERFC_Correlation_Factor::calculate_f12_c(double r12) {
  return exp(-gamma * gamma * r12 * r12);
}
double ERFC_Correlation_Factor::calculate_f12_d(double r12) { 
  return 0.0;
}
bool ERFC_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double ERFC_Linear_Correlation_Factor::calculate_f12(double r12) {
  return r12 * erfc(gamma * r12);
}
double ERFC_Linear_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * erfc(gamma * r12);
}
double ERFC_Linear_Correlation_Factor::calculate_f12_b(double r12) {
  auto d = gamma * gamma * r12 * r12;
  return exp(-d) * (8.0 * gamma - 4.0 * gamma * d) / sqrt_pi;
}
double ERFC_Linear_Correlation_Factor::calculate_f12_c(double r12) {
  return erfc(gamma * r12);
}
double ERFC_Linear_Correlation_Factor::calculate_f12_d(double r12) {
  return -2.0 * exp(-gamma * gamma * r12 * r12) * gamma / sqrt_pi;
}
bool ERFC_Linear_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Tanh_Correlation_Factor::calculate_f12(double r12) {
  return tanh(gamma * r12) / gamma;
}
double Tanh_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 / (cosh(gamma * r12) * cosh(gamma * r12));
}
double Tanh_Correlation_Factor::calculate_f12_b(double r12) {
  return 2.0 * gamma * tanh(gamma * r12) / (cosh(gamma * r12) * cosh(gamma * r12));
}
double Tanh_Correlation_Factor::calculate_f12_c(double r12) {
  return 1 / (cosh(gamma * r12) * cosh(gamma * r12));
}
double Tanh_Correlation_Factor::calculate_f12_d(double r12) { 
  return 0.0; 
}
bool Tanh_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double ArcTan_Correlation_Factor::calculate_f12(double r12) {
  return atan(gamma * r12) / gamma;
}
double ArcTan_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = 1.0 + gamma * gamma * r12 * r12;
  return -2.0 / (d1 * d1);
}
double ArcTan_Correlation_Factor::calculate_f12_b(double r12) { 
  return 0.0;
}
double ArcTan_Correlation_Factor::calculate_f12_c(double r12) {
  return 1.0 / (1.0 + gamma * gamma * r12 * r12);
}
double ArcTan_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0; 
}
bool ArcTan_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Logarithm_Correlation_Factor::calculate_f12(double r12) {
  return log(1.0 + gamma * r12) / gamma;
}
double Logarithm_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = 1.0 + gamma * r12;
  return -2.0 / (d1 * d1);
}
double Logarithm_Correlation_Factor::calculate_f12_b(double r12) {
  double d1 = 1.0 + gamma * r12;
  return -gamma / (d1 * d1);
}
double Logarithm_Correlation_Factor::calculate_f12_c(double r12) {
  return 1.0 / (1.0 + gamma * r12);
}
double Logarithm_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Logarithm_Correlation_Factor::f12_d_is_zero() {
  return true;
}


double Hybrid_Correlation_Factor::calculate_f12(double r12) {
  return -0.5 * exp(-gamma * r12) / gamma + 0.5 * gamma * r12 / (gamma + r12);
}
double Hybrid_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  return -exp(-gamma * r12) - d1 * d1 * d1;
}
double Hybrid_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.5 * gamma * exp(-gamma * r12);
}
double Hybrid_Correlation_Factor::calculate_f12_c(double r12) {
  return 0.5 * (exp(-gamma * r12) + gamma / (gamma + r12));
}
double Hybrid_Correlation_Factor::calculate_f12_d(double r12) {
  return -0.5 * gamma / ((gamma + r12) * (gamma + r12));
}
bool Hybrid_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Two_Parameter_Rational_Correlation_Factor::calculate_f12(double r12) {
  return gamma * r12 / (gamma + beta * r12);
}
double Two_Parameter_Rational_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = gamma / (gamma + beta * r12);
  return -2.0 * d1 * d1 * d1;
}
double Two_Parameter_Rational_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double Two_Parameter_Rational_Correlation_Factor::calculate_f12_c(double r12) {
  return gamma / (gamma + beta * r12);
}
double Two_Parameter_Rational_Correlation_Factor::calculate_f12_d(double r12) {
  double d1 = gamma + beta * r12;
  return -beta * gamma / (d1 * d1);
}
bool Two_Parameter_Rational_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Higher_Rational_Correlation_Factor::calculate_f12(double r12) {
  double denominator = beta + r12;
  return gamma * r12 / (2.0 * (gamma + r12)) + beta * beta * r12 / (2.0 * (denominator * denominator));
}
double Higher_Rational_Correlation_Factor::calculate_f12_a(double r12) {
  double d1 = gamma / (gamma + r12);
  double d2 = beta / (beta + r12);
  return -d1 * d1 * d1 + (2 * r12 - beta) * (d2 * d2 * d2 * d2) / beta;
}
double Higher_Rational_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double Higher_Rational_Correlation_Factor::calculate_f12_c(double r12) {
  return 0.5 *
         (gamma / (gamma + r12) + beta * beta / ((beta + r12) * (beta + r12)));
}
double Higher_Rational_Correlation_Factor::calculate_f12_d(double r12) {
  double d1 = gamma / (gamma + r12);
  double d2 = beta / (beta + r12);
  return -0.5 * d1 * d1 / gamma - d2 * d2 * d2 / beta;
}
bool Higher_Rational_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Cubic_Slater_Correlation_Factor::calculate_f12(double r12) {
  // return (1.0-exp(-gamma*r12))/gamma+exp(-beta*r12*r12*r12)-1;
  return (-exp(-gamma * r12)) / gamma + exp(-beta * r12 * r12 * r12);
}
double Cubic_Slater_Correlation_Factor::calculate_f12_a(double r12) {
  return -2.0 * exp(-gamma * r12);
}
double Cubic_Slater_Correlation_Factor::calculate_f12_b(double r12) {
  double er12 = exp(-gamma * r12);
  double r12_3 = r12 * r12 * r12;
  double er12_3 = exp(-beta * r12_3);
  return gamma * er12 + r12 * er12_3 * beta * (12.0 - 9.0 * r12_3 * beta);
}
double Cubic_Slater_Correlation_Factor::calculate_f12_c(double r12) {
  return exp(-gamma * r12);
}
double Cubic_Slater_Correlation_Factor::calculate_f12_d(double r12) {
  double r12_3 = r12 * r12 * r12;
  return -3.0 * beta * r12 * exp(-beta * r12_3);
}
bool Cubic_Slater_Correlation_Factor::f12_d_is_zero() {
  return false;
}


double Higher_Jastrow_Correlation_Factor::calculate_f12(double r12) {
  double fac1 = (1 - exp(-beta * r12)) / beta;
  return exp(fac1 / (1 + gamma * fac1));
}
double Higher_Jastrow_Correlation_Factor::calculate_f12_a(double r12) {
  double er12 = exp(beta * r12);
  double fac1 = beta + gamma;
  double fac2 = beta * beta * exp(1 / fac1 + beta * (r12 + 1 / (gamma * fac1 - fac1 * fac1 * er12)));
  double fac3 = gamma - fac1 * er12;
  return fac2 * (-(2.0 + beta * r12) * gamma * gamma +
           exp(2 * beta * r12) * (beta * r12 - 2.0) * fac1 * fac1 +
           er12 * (4.0 * gamma * (fac1)-beta * beta * r12)) /
          (fac3 * fac3 * fac3 * fac3);
}
double Higher_Jastrow_Correlation_Factor::calculate_f12_b(double r12) {
  return 0.0;
}
double Higher_Jastrow_Correlation_Factor::calculate_f12_c(double r12) {
  double er12 = exp(beta * r12);
  double fac1 = beta + gamma;
  double fac2 = beta * beta * exp(1 / fac1 + beta * (r12 + 1 / (gamma * fac1 - fac1 * fac1 * er12)));
  double fac3 = gamma - fac1 * er12;
  return fac2 / (fac3 * fac3);
}
double Higher_Jastrow_Correlation_Factor::calculate_f12_d(double r12) {
  return 0.0;
}
bool Higher_Jastrow_Correlation_Factor::f12_d_is_zero() {
  return true;
}

//
// Created by aedoran on 12/18/19.
//

#include "correlation_factor.h"
#include "correlation_factor_function.h"

Correlation_Factor* create_correlation_factor(CORRELATION_FACTOR::Type correlation_factor_id, double gamma, double beta) {
  Correlation_Factor* correlation_factor = nullptr;
  switch (correlation_factor_id) {
    case CORRELATION_FACTOR::Linear: correlation_factor =  new Linear_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Rational: correlation_factor =  new Rational_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Slater: correlation_factor =  new Slater_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Slater_Linear: correlation_factor =  new Slater_Linear_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Gaussian: correlation_factor =  new Gaussian_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Cusped_Gaussian: correlation_factor =  new Cusped_Gaussian_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Yukawa_Coulomb: correlation_factor =  new Yukawa_Coulomb_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Jastrow: correlation_factor =  new Jastrow_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::ERFC: correlation_factor =  new ERFC_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::ERFC_Linear: correlation_factor =  new ERFC_Linear_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Tanh: correlation_factor =  new Tanh_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::ArcTan: correlation_factor =  new ArcTan_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Logarithm: correlation_factor =  new Logarithm_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Hybrid: correlation_factor =  new Hybrid_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Two_Parameter_Rational: correlation_factor =  new Two_Parameter_Rational_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Higher_Rational: correlation_factor =  new Higher_Rational_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Cubic_Slater: correlation_factor =  new Cubic_Slater_Correlation_Factor(gamma, beta); break;
    case CORRELATION_FACTOR::Higher_Jastrow: correlation_factor =  new Higher_Jastrow_Correlation_Factor(gamma, beta); break;
  }
  return correlation_factor;
}

Correlation_Factor::Correlation_Factor(double gamma_in, double beta_in) :
    m_gamma(gamma_in), 
    m_beta(beta_in) {
}

Correlation_Factor::Correlation_Factor(double gamma_in, double beta_in, double default_gamma, double default_beta) {
  m_gamma = gamma_in;
  if (gamma_in < 0) {
    m_gamma = default_gamma;
  }
  m_beta = beta_in;
  if (beta_in < 0) {
    m_beta = default_beta;
  }
}

double Correlation_Factor::gamma() {
  return m_gamma;
}

double Correlation_Factor::beta() {
  return m_beta;
}

#define SOURCE_FILE "correlation_factor.imp.cpp"
#include "correlation_factor_patterns.h"
#undef SOURCE_FILE

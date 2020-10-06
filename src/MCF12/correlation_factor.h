#ifndef CORRELATION_FACTOR_H_
#define CORRELATION_FACTOR_H_

#include "correlation_factor_types.h"

class Correlation_Factor {
 public:
  Correlation_Factor(double gamma_in, double beta_in);
  Correlation_Factor(double gamma_in, double beta_in, double default_gamma, double default_beta);
  virtual double f12(double r12) = 0;
  virtual double f12_a(double r12) = 0;
  virtual double f12_b(double r12) = 0;
  virtual double f12_c(double r12) = 0;
  virtual double f12_d(double r12) = 0;
  virtual bool f12_d_is_zero() = 0;
  double gamma();
  double beta();

 protected:
  double m_gamma;
  double m_beta;
};

Correlation_Factor* create_correlation_factor(CORRELATION_FACTOR::Type, double gamma = -1, double beta = -1);

#define SOURCE_FILE "correlation_factor.imp.h"
#include "correlation_factor_patterns.h"
#undef SOURCE_FILE

#endif // CORRELATION_FACTOR_H_


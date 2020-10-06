#ifndef CORRELATION_FACTOR_TYPES_H_
#define CORRELATION_FACTOR_TYPES_H_

#include <string>

namespace CORRELATION_FACTOR {
  enum Type {
    Linear,
    Rational,
    Slater,
    Slater_Linear,
    Gaussian,
    Cusped_Gaussian,
    Yukawa_Coulomb,
    Jastrow,
    ERFC,
    ERFC_Linear,
    Tanh,
    ArcTan,
    Logarithm,
    Hybrid,
    Two_Parameter_Rational,
    Higher_Rational,
    Cubic_Slater,
    Higher_Jastrow,
  };
}

CORRELATION_FACTOR::Type string_to_correlation_factors(const std::string&);
std::string correlation_factors_to_string(const CORRELATION_FACTOR::Type&);

#endif // CORRELATION_FACTOR_TYPES_H_

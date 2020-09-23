#ifndef CORRELATION_FACTORS_H_
#define CORRELATION_FACTORS_H_

namespace CORRELATION_FACTORS {
  enum CORRELATION_FACTORS {
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

CORRELATION_FACTORS::CORRELATION_FACTORS string_to_correlation_factors(const std::string&);

std::string correlation_factors_to_string(const CORRELATION_FACTORS::CORRELATION_FACTORS&);

#endif // CORRELATION_FACTORS_H_

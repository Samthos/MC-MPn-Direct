#include <iostream>
#include <string>

#include "correlation_factor.h"

CORRELATION_FACTOR::Type string_to_correlation_factors(const std::string& str) {
  CORRELATION_FACTOR::Type correlation_factor;
  if (str == "Linear") {
    correlation_factor = CORRELATION_FACTOR::Linear;
  } else if (str == "Rational") {
    correlation_factor = CORRELATION_FACTOR::Rational;
  } else if (str == "Slater") {
    correlation_factor = CORRELATION_FACTOR::Slater;
  } else if (str == "Slater_Linear") {
    correlation_factor = CORRELATION_FACTOR::Slater_Linear;
  } else if (str == "Gaussian") {
    correlation_factor = CORRELATION_FACTOR::Gaussian;
  } else if (str == "Cusped_Gaussian") {
    correlation_factor = CORRELATION_FACTOR::Cusped_Gaussian;
  } else if (str == "Yukawa_Coulomb") {
    correlation_factor = CORRELATION_FACTOR::Yukawa_Coulomb;
  } else if (str == "Jastrow") {
    correlation_factor = CORRELATION_FACTOR::Jastrow;
  } else if (str == "ERFC") {
    correlation_factor = CORRELATION_FACTOR::ERFC;
  } else if (str == "ERFC_Linear") {
    correlation_factor = CORRELATION_FACTOR::ERFC_Linear;
  } else if (str == "Tanh") {
    correlation_factor = CORRELATION_FACTOR::Tanh;
  } else if (str == "ArcTan") {
    correlation_factor = CORRELATION_FACTOR::ArcTan;
  } else if (str == "Logarithm") {
    correlation_factor = CORRELATION_FACTOR::Logarithm;
  } else if (str == "Hybrid") {
    correlation_factor = CORRELATION_FACTOR::Hybrid;
  } else if (str == "Two_Parameter_Rational") {
    correlation_factor = CORRELATION_FACTOR::Two_Parameter_Rational;
  } else if (str == "Higher_Rational") {
    correlation_factor = CORRELATION_FACTOR::Higher_Rational;
  } else if (str == "Cubic_Slater") {
    correlation_factor = CORRELATION_FACTOR::Cubic_Slater;
  } else if (str == "Higher_Jastrow") {
    correlation_factor = CORRELATION_FACTOR::Higher_Jastrow;
  } else {
    std::cerr << "Correlation factor " << str << " not supported\n";
    exit(0);
  }
  return correlation_factor;
}
std::string correlation_factors_to_string(const CORRELATION_FACTOR::Type& correlation_factor) {
  std::string str;
  switch (correlation_factor) {
    case CORRELATION_FACTOR::Linear: str = "Linear"; break;
    case CORRELATION_FACTOR::Rational: str = "Rational"; break;
    case CORRELATION_FACTOR::Slater: str = "Slater"; break;
    case CORRELATION_FACTOR::Slater_Linear: str = "Slater_Linear"; break;
    case CORRELATION_FACTOR::Gaussian: str = "Gaussian"; break;
    case CORRELATION_FACTOR::Cusped_Gaussian: str = "Cusped_Gaussian"; break;
    case CORRELATION_FACTOR::Yukawa_Coulomb: str = "Yukawa_Coulomb"; break;
    case CORRELATION_FACTOR::Jastrow: str = "Jastrow"; break;
    case CORRELATION_FACTOR::ERFC: str = "ERFC"; break;
    case CORRELATION_FACTOR::ERFC_Linear: str = "ERFC_Linear"; break;
    case CORRELATION_FACTOR::Tanh: str = "Tanh"; break;
    case CORRELATION_FACTOR::ArcTan: str = "ArcTan"; break;
    case CORRELATION_FACTOR::Logarithm: str = "Logarithm"; break;
    case CORRELATION_FACTOR::Hybrid: str = "Hybrid"; break;
    case CORRELATION_FACTOR::Two_Parameter_Rational: str = "Two_Parameter_Rational"; break;
    case CORRELATION_FACTOR::Higher_Rational: str = "Higher_Rational"; break;
    case CORRELATION_FACTOR::Cubic_Slater: str = "Cubic_Slater"; break;
    case CORRELATION_FACTOR::Higher_Jastrow: str = "Higher_Jastrow"; break;
  }
  return str;
}

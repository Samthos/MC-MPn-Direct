#include <iostream>
#include <string>

#include "correlation_factor.h"

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

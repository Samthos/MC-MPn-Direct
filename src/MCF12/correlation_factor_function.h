#ifndef CORRELATION_FACTOR_FUNCTION_H_
#define CORRELATION_FACTOR_FUNCTION_H_

#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#include <memory>
#include <vector>

#include "correlation_factor.h"

class Correlation_Factor_Function {
 public:
  HOSTDEVICE Correlation_Factor_Function(double gamma_, double beta_);
  HOSTDEVICE Correlation_Factor_Function(double gamma_, double beta_, double default_gamma, double default_beta);
  virtual HOSTDEVICE HOSTDEVICE double f12(double r12) = 0;
  virtual HOSTDEVICE HOSTDEVICE double f12_a(double r12) = 0;
  virtual HOSTDEVICE HOSTDEVICE double f12_b(double r12) = 0;
  virtual HOSTDEVICE HOSTDEVICE double f12_c(double r12) = 0;
  virtual HOSTDEVICE HOSTDEVICE double f12_d(double r12) = 0;
  virtual bool f12_d_is_zero() = 0;

 protected:
  double gamma, beta;
};
Correlation_Factor_Function* create_correlation_factor_function(CORRELATION_FACTORS::CORRELATION_FACTORS, double gamma = -1, double beta = -1);

class Linear_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Linear_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {};
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;

 private:
  static constexpr double default_gamma = 0.0;
  static constexpr double default_beta  = 0.0;
};
class Rational_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Rational_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Slater_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Slater_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  bool f12_d_is_zero() override;
 private:
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Slater_Linear_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Slater_Linear_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.5;
  static constexpr double default_beta  = 0.0;
};
class Gaussian_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Gaussian_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.5;
  static constexpr double default_beta  = 0.0;
};
class Cusped_Gaussian_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Cusped_Gaussian_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Yukawa_Coulomb_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Yukawa_Coulomb_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 2.0;
  static constexpr double default_beta  = 0.0;
};
class Jastrow_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Jastrow_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ERFC_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  ERFC_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ERFC_Linear_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  ERFC_Linear_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
 private:
  static constexpr double default_gamma = 0.4;
  static constexpr double default_beta  = 0.0;
};
class Tanh_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Tanh_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ArcTan_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  ArcTan_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.6;
  static constexpr double default_beta  = 0.0;
};
class Logarithm_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Logarithm_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 2.0;
  static constexpr double default_beta  = 0.0;
};
class Hybrid_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Hybrid_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Two_Parameter_Rational_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Two_Parameter_Rational_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.0;
  static constexpr double default_beta  = 1.0;
};
class Higher_Rational_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Higher_Rational_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.6;
  static constexpr double default_beta  = 3.0;
};
class Cubic_Slater_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Cubic_Slater_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.003;
};
class Higher_Jastrow_Correlation_Factor_Function : public Correlation_Factor_Function {
 public:
  Higher_Jastrow_Correlation_Factor_Function(double gamma, double beta) : Correlation_Factor_Function(gamma, beta, default_gamma, default_beta) {}
  HOSTDEVICE double f12(double r12) override;
  HOSTDEVICE double f12_a(double r12) override;
  HOSTDEVICE double f12_b(double r12) override;
  HOSTDEVICE double f12_c(double r12) override;
  HOSTDEVICE double f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.8;
  static constexpr double default_beta  = 0.75;
};

#undef HOSTDEVICE
#endif // CORRELATION_FACTOR_FUNCTION_H_

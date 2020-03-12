//
// Created by aedoran on 12/31/19.
//

#ifndef F12_METHODS_SRC_CORRELATION_FACTORS_H_
#define F12_METHODS_SRC_CORRELATION_FACTORS_H_

#include <vector>

#include "../qc_input.h"
#include "../electron_list.h"
#include "../electron_pair_list.h"

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

class Correlation_Factor {
 public:
  Correlation_Factor(const IOPs& iops, double gamma_, double beta_) :
      one_e_r12(iops.iopns[KEYS::ELECTRONS], std::vector<double>(iops.iopns[KEYS::ELECTRONS], 0.0)),
      f12o(iops.iopns[KEYS::ELECTRONS], std::vector<double>(iops.iopns[KEYS::ELECTRONS], 0.0)),
      f13p(iops.iopns[KEYS::ELECTRON_PAIRS], std::vector<double>(iops.iopns[KEYS::ELECTRONS], 0.0)),
      f23p(iops.iopns[KEYS::ELECTRON_PAIRS], std::vector<double>(iops.iopns[KEYS::ELECTRONS], 0.0)),
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
  virtual double calculate_f12(double r12) = 0;
  virtual double calculate_f12_a(double r12) = 0;
  virtual double calculate_f12_b(double r12) = 0;
  virtual double calculate_f12_c(double r12) = 0;
  virtual double calculate_f12_d(double r12) = 0;
  virtual bool f12_d_is_zero() = 0;

  void update(const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list);

  std::vector<std::vector<double>> one_e_r12;
  std::vector<std::vector<double>> f12o;
  std::vector<std::vector<double>> f13p;
  std::vector<std::vector<double>> f23p;

 protected:
  double gamma, beta;
  static double distance(const std::array<double, 3>&, const std::array<double, 3>&);
};
Correlation_Factor* create_correlation_factor(const IOPs& iops);
CORRELATION_FACTORS::CORRELATION_FACTORS string_to_correlation_factors(const std::string&);
std::string correlation_factors_to_string(const CORRELATION_FACTORS::CORRELATION_FACTORS&);

class Linear_Correlation_Factor : public Correlation_Factor {
 public:
  Linear_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {};
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.0;
  static constexpr double default_beta  = 0.0;
};
class Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Rational_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Slater_Correlation_Factor : public Correlation_Factor {
 public:
  Slater_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Slater_Linear_Correlation_Factor : public Correlation_Factor {
 public:
  Slater_Linear_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.5;
  static constexpr double default_beta  = 0.0;
};
class Gaussian_Correlation_Factor : public Correlation_Factor {
 public:
  Gaussian_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.5;
  static constexpr double default_beta  = 0.0;
};
class Cusped_Gaussian_Correlation_Factor : public Correlation_Factor {
 public:
  Cusped_Gaussian_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Yukawa_Coulomb_Correlation_Factor : public Correlation_Factor {
 public:
  Yukawa_Coulomb_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 2.0;
  static constexpr double default_beta  = 0.0;
};
class Jastrow_Correlation_Factor : public Correlation_Factor {
 public:
  Jastrow_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ERFC_Correlation_Factor : public Correlation_Factor {
 public:
  ERFC_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ERFC_Linear_Correlation_Factor : public Correlation_Factor {
 public:
  ERFC_Linear_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
 private:
  static constexpr double default_gamma = 0.4;
  static constexpr double default_beta  = 0.0;
};
class Tanh_Correlation_Factor : public Correlation_Factor {
 public:
  Tanh_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class ArcTan_Correlation_Factor : public Correlation_Factor {
 public:
  ArcTan_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.6;
  static constexpr double default_beta  = 0.0;
};
class Logarithm_Correlation_Factor : public Correlation_Factor {
 public:
  Logarithm_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 2.0;
  static constexpr double default_beta  = 0.0;
};
class Hybrid_Correlation_Factor : public Correlation_Factor {
 public:
  Hybrid_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.0;
};
class Two_Parameter_Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Two_Parameter_Rational_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.0;
  static constexpr double default_beta  = 1.0;
};
class Higher_Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Higher_Rational_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.6;
  static constexpr double default_beta  = 3.0;
};
class Cubic_Slater_Correlation_Factor : public Correlation_Factor {
 public:
  Cubic_Slater_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 1.2;
  static constexpr double default_beta  = 0.003;
};
class Higher_Jastrow_Correlation_Factor : public Correlation_Factor {
 public:
  Higher_Jastrow_Correlation_Factor(const IOPs& iops) : Correlation_Factor(iops, default_gamma, default_beta) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  static constexpr double default_gamma = 0.8;
  static constexpr double default_beta  = 0.75;
};

#endif //F12_METHODS_SRC_CORRELATION_FACTORS_H_

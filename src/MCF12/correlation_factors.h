//
// Created by aedoran on 12/31/19.
//

#ifndef F12_METHODS_SRC_CORRELATION_FACTORS_H_
#define F12_METHODS_SRC_CORRELATION_FACTORS_H_
#include <vector>
#include "../electron_list.h"
#include "../electron_pair_list.h"

class Correlation_Factor {
 public:
  Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) :
      one_e_r12(electrons, std::vector<double>(electrons, 0.0)),
      f12o(electrons, std::vector<double>(electrons, 0.0)),
      f13p(electron_pairs, std::vector<double>(electrons, 0.0)),
      f23p(electron_pairs, std::vector<double>(electrons, 0.0)),
      gamma(gamma_), beta(beta_) {}
  virtual double calculate_f12(double r12) = 0;
  virtual double calculate_f12_a(double r12) = 0;
  virtual double calculate_f12_b(double r12) = 0;
  virtual double calculate_f12_c(double r12) = 0;
  virtual double calculate_f12_d(double r12) = 0;
  virtual bool f12_d_is_zero() = 0;

  void update(const Electron_Pair_List& electron_pair_list, const Electron_List& electron_list);

  std::vector<std::vector<double>> one_e_r12;
  std::vector<std::vector<double>> f12o;
  std::vector<std::vector<double>> f13p;
  std::vector<std::vector<double>> f23p;

 protected:
  double gamma, beta;
};
Correlation_Factor* make_correlation_factor(int electron_pairs, int electrons, int f12_corr_id, double gamma, double beta);

class Linear_Correlation_Factor : public Correlation_Factor {
 public:
  Linear_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {};
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Quadratic_Correlation_Factor : public Correlation_Factor {
 public:
  Quadratic_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Rational_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Slater_Correlation_Factor : public Correlation_Factor {
 public:
  Slater_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Slater_Linear_Correlation_Factor : public Correlation_Factor {
 public:
  Slater_Linear_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Gaussian_Correlation_Factor : public Correlation_Factor {
 public:
  Gaussian_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Cusped_Gaussian_Correlation_Factor : public Correlation_Factor {
 public:
  Cusped_Gaussian_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Yukawa_Coulomb_Correlation_Factor : public Correlation_Factor {
 public:
  Yukawa_Coulomb_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Jastrow_Correlation_Factor : public Correlation_Factor {
 public:
  Jastrow_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class ERFC_Correlation_Factor : public Correlation_Factor {
 public:
  ERFC_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
};
class ERFC_Linear_Correlation_Factor : public Correlation_Factor {
 public:
  ERFC_Linear_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
 private:
  constexpr static double sqrt_pi = 1.7724538509055160273;
};
class Sinh_Correlation_Factor : public Correlation_Factor {
 public:
  Sinh_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Tanh_Correlation_Factor : public Correlation_Factor {
 public:
  Tanh_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class ArcTan_Correlation_Factor : public Correlation_Factor {
 public:
  ArcTan_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Logarithm_Correlation_Factor : public Correlation_Factor {
 public:
  Logarithm_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Hybrid_Correlation_Factor : public Correlation_Factor {
 public:
  Hybrid_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Cubic_Correlation_Factor : public Correlation_Factor {
 public:
  Cubic_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Two_Parameter_Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Two_Parameter_Rational_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Higher_Rational_Correlation_Factor : public Correlation_Factor {
 public:
  Higher_Rational_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Cubic_Slater_Correlation_Factor : public Correlation_Factor {
 public:
  Cubic_Slater_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};
class Higher_Jastrow_Correlation_Factor : public Correlation_Factor {
 public:
  Higher_Jastrow_Correlation_Factor(int electron_pairs, int electrons, double gamma_, double beta_) : Correlation_Factor(electron_pairs, electrons, gamma_, beta_) {}
  double calculate_f12(double r12) override;
  double calculate_f12_a(double r12) override;
  double calculate_f12_b(double r12) override;
  double calculate_f12_c(double r12) override;
  double calculate_f12_d(double r12) override;
  bool f12_d_is_zero() override;
};

#endif //F12_METHODS_SRC_CORRELATION_FACTORS_H_

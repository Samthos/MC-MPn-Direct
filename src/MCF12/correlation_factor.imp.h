class FUNCTIONAL_NAME(_Correlation_Factor) : public Correlation_Factor {
 public:
  FUNCTIONAL_NAME(_Correlation_Factor)(double gamma, double beta) : Correlation_Factor(gamma, beta, default_gamma, default_beta) {};
  double f12(double r12) override;
  double f12_a(double r12) override;
  double f12_b(double r12) override;
  double f12_c(double r12) override;
  double f12_d(double r12) override;
  bool f12_d_is_zero() override;

 private:
  static constexpr double default_gamma = DEFAULT_GAMMA;
  static constexpr double default_beta  = DEFAULT_BETA ;
};

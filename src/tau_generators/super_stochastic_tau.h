#ifndef SUPER_STOCHASTIC_TAU_H_
#define SUPER_STOCHASTIC_TAU_H_

#include "tau.h"

class Super_Stochastic_Tau : public Tau {
 public:
  explicit Super_Stochastic_Tau(const std::shared_ptr<Movec_Parser> basis);
  ~Super_Stochastic_Tau() = default;

  void resize(int dimm) override;
  void new_tau(Random& random) override;
  std::vector<double> get_exp_tau(int stop, int start) override;
  double get_gfn_tau(int stop, int start, int offset, int conjugate) override;
  double get_wgt(int dimm) override;
  double get_tau(int index) override;
  size_t get_n_coordinates() override;
  bool next() override;
  bool is_new(int i) override;
  void set_from_other(Tau* other);

 private:
  void set_pdf_cdf(std::vector<double>& pdf, std::vector<double>& cdf, int first, int last);
  void set_lambda(int index, Random& random);
  static int choose_index(Random& random, std::vector<double>& cdf, int offset);
  void set_weight(int index);

  std::vector<double> lambda;
  std::vector<double> hole_pdf, hole_cdf;
  std::vector<double> particle_pdf, particle_cdf;
};

#endif  // SUPER_STOCHASTIC_TAU_H_


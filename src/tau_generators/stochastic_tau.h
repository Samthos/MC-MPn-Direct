#ifndef STOCHASTIC_TAU_H_
#define STOCHASTIC_TAU_H_

#include "tau.h"

class Stochastic_Tau : public Tau {
 public:
  explicit Stochastic_Tau(const std::shared_ptr<Movec_Parser> basis);
  ~Stochastic_Tau() = default;

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
  void set(const std::vector<double>&);

 private:
  void update();
  double lambda;
};

#endif  // STOCHASTIC_TAU_H_

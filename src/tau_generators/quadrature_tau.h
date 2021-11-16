#ifndef QUADRATURE_TAU_H_
#define QUADRATURE_TAU_H_

#include "tau.h"

class Quadrature_Tau : public Tau {
 public:
  Quadrature_Tau(const std::shared_ptr<Movec_Parser> basis);

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
  std::vector<int> indices;
};

#endif  // QUADRATURE_TAU_H_

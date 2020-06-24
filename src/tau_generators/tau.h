//
// Created by aedoran on 6/8/18.
//

#ifndef TAU_H_
#define TAU_H_

#include <algorithm>
#include <iostream>
#include <vector>

#include "../basis/nwchem_movec_parser.h"
#include "../qc_random.h"
#include "../qc_input.h"

class Tau {
 public:
  explicit Tau(const std::shared_ptr<Movec_Parser> basis);
  virtual void resize(int dimm) = 0;
  virtual void new_tau(Random& random) = 0;
  virtual std::vector<double> get_exp_tau(int, int) = 0;
  virtual double get_gfn_tau(int, int, int ,int) = 0;
  virtual double get_wgt(int) = 0;
  virtual double get_tau(int) = 0;
  virtual size_t get_n_coordinates() = 0;
  virtual bool next() = 0;
  virtual bool is_new(int) = 0;
  virtual void set_from_other(Tau*) = 0;

  void copy_p(Tau* other);

 protected:
  int iocc1, iocc2, ivir1, ivir2;
  std::vector<double> p;
  std::vector<double> evals;
  std::vector<double> tau;
  std::vector<double> wgt;
  std::vector<double> scratch;
  std::vector<std::vector<double>> exp_tau;
};

Tau* create_tau_sampler(const IOPs& iops, const std::shared_ptr<Movec_Parser> basis);
#endif // TAU_H_

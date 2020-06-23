//
// Created by aedoran on 6/8/18.
//

#ifndef MC_MP3_DIRECT_TAU_INTEGRALS_H
#define MC_MP3_DIRECT_TAU_INTEGRALS_H

#include <algorithm>
#include <iostream>
#include <vector>

#include "basis/nwchem_movec_parser.h"
#include "qc_random.h"

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

 private:
  void update();
  double lambda;
};

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

Tau* create_tau_sampler(const IOPs& iops, const std::shared_ptr<Movec_Parser> basis);
#endif //MC_MP3_DIRECT_TAU_INTEGRALS_H

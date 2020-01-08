// Copyright 2017 Hirata Lab

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <array>
#include <vector>

#include "weight_function.h"
#include "qc_geom.h"
#include "qc_random.h"
#include "qc_input.h"

#ifndef EL_PAIR_H_
#define EL_PAIR_H_
struct Electron_Pair {
  std::array<double, 3> pos1, pos2;
  double wgt, rv;
};
std::ostream& operator << (std::ostream& os, const Electron_Pair& electron_pair);

class Electron_Pair_List {
 public:
  explicit Electron_Pair_List(int size);
  virtual ~Electron_Pair_List() = default;
  virtual void move(Random&, const Molec&, const Electron_Pair_GTO_Weight&) = 0;
  virtual bool requires_blocking() = 0;

  // functions to emulate vector interface
  std::size_t size() {
    return electron_pairs.size();
  }

  std::vector<std::array<double, 3>> pos1;
  std::vector<std::array<double, 3>> pos2;
  std::vector<double> wgt;
  std::vector<double> rv;
 protected:
  static double calculate_r12(const Electron_Pair &electron_pair);
  static void set_weight(Electron_Pair&, const Electron_Pair_GTO_Weight&);
  void transpose();
  std::vector<Electron_Pair>::iterator begin() {
    return electron_pairs.begin();
  }
  std::vector<Electron_Pair>::iterator end() {
    return electron_pairs.end();
  }


  std::vector<Electron_Pair> electron_pairs;
};
Electron_Pair_List* create_sampler(IOPs& iops, Molec& molec, Electron_Pair_GTO_Weight& weight);

class Direct_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Direct_Electron_Pair_List(int size) : Electron_Pair_List(size) {}
  ~Direct_Electron_Pair_List() override = default;
  void move(Random& random, const Molec& molec, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void mc_move_scheme(Electron_Pair&, Random&, const Molec&, const Electron_Pair_GTO_Weight&);
  static double calculate_r(Random& random, double alpha, double beta, double a);
  static double calculate_phi(double p, double r, double alpha, double beta, double a);
  static double CDF(const double& rho, const double& c, const double& erf_c);
  static double PDF(const double& rho, const double& c, const double& erf_c);
  static double PDF_Prime(const double& rho, const double& c, const double& erf_c);
};

class Metropolis_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Metropolis_Electron_Pair_List(int size, double ml, Random& random, const Molec& molec, const Electron_Pair_GTO_Weight& weight);
  ~Metropolis_Electron_Pair_List() override = default;
  void move(Random& random, const Molec& molec, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void initialize(Electron_Pair&, Random&, const Molec&, const Electron_Pair_GTO_Weight&);
  void mc_move_scheme(Electron_Pair&, Random&, const Molec&, const Electron_Pair_GTO_Weight&);
  void rescale_move_length();

  double move_length;
  int moves_since_rescale;
  int successful_moves;
  int failed_moves;
};
#endif  // EL_PAIR_H_

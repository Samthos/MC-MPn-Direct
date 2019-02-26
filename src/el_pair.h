// Copyright 2017 Hirata Lab

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <array>
#include <vector>

#include "weight_function.h"
#include "qc_geom.h"
#include "qc_random.h"

#ifndef EL_PAIR_H_
#define EL_PAIR_H_
struct Electron_Pair {
  std::array<double, 3> pos1, pos2;
  double wgt, rv;
};

class Electron_Pair_List {
 public:
  explicit Electron_Pair_List(int size) : electron_pairs(size) {}
  virtual void move(Random&, const Molec&, const GTO_Weight&) = 0;

  // functions to emulate vector interface
  std::size_t size() {
    return electron_pairs.size();
  }
  Electron_Pair* data() {
    return electron_pairs.data();
  }
  const Electron_Pair& operator [] (int i) const {
    return electron_pairs[i];
  }
  std::vector<Electron_Pair>::iterator begin() {
    return electron_pairs.begin();
  }
  std::vector<Electron_Pair>::iterator end() {
    return electron_pairs.end();
  }

 protected:
  static double calculate_r12(const Electron_Pair &el_pair);
  static void set_weight(Electron_Pair&, const GTO_Weight&);
  std::vector<Electron_Pair> electron_pairs;
};

class Direct_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Direct_Electron_Pair_List(int size) : Electron_Pair_List(size) {}
  void move(Random& random, const Molec& molec, const GTO_Weight& weight) override {
    for (Electron_Pair &electron_pair : electron_pairs) {
      mc_move_scheme(electron_pair, random, molec, weight);
    }
  }
 private:
  static void mc_move_scheme(Electron_Pair&, Random&, const Molec&, const GTO_Weight&);
  static double calculate_r(Random& random, double alpha, double beta, double a);
  static double calculate_phi(double p, double r, double alpha, double beta, double a);
  static double CDF(const double& rho, const double& c, const double& erf_c);
  static double PDF(const double& rho, const double& c, const double& erf_c);
  static double PDF_Prime(const double& rho, const double& c, const double& erf_c);
};

class Metropolis_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Metropolis_Electron_Pair_List(int size, double ml, Random& random, const Molec& molec, const GTO_Weight& weight) : Electron_Pair_List(size),
      move_length(ml), moves_since_rescale(0), successful_moves(0), failed_moves(0){
    // initilizie pos
    for (Electron_Pair& electron_pair : electron_pairs) {
      initialize(electron_pair, random, molec, weight);
    }
    // burn in
    for (int i = 0; i < 100'000; i++) {
      move(random, molec, weight);
    }
  }
  void move(Random& random, const Molec& molec, const GTO_Weight& weight) override {
    if (moves_since_rescale == 1'000) {
      rescale_move_length();
    }
    for (Electron_Pair &electron_pair : electron_pairs) {
      mc_move_scheme(electron_pair, random, molec, weight);
    }
    moves_since_rescale++;
  }
 private:
  static void initialize(Electron_Pair&, Random&, const Molec&, const GTO_Weight&);
  void mc_move_scheme(Electron_Pair&, Random&, const Molec&, const GTO_Weight&);
  void rescale_move_length();

  double move_length;
  int moves_since_rescale;
  int successful_moves;
  int failed_moves;
};
#endif  // EL_PAIR_H_

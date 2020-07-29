#ifndef ELECTRON_PAIR_LIST_H_
#define ELECTRON_PAIR_LIST_H_
#include <array>
#include <vector>

#include "../weight_function.h"
#include "molecule.h"
#include "../qc_random.h"
#include "../qc_input.h"

struct Electron_Pair {
  Point pos1, pos2;
  double wgt;
  double rv;
  double r12;
};
std::ostream& operator << (std::ostream& os, const Electron_Pair& electron_pair);

class Electron_Pair_List {
 public:
  explicit Electron_Pair_List(int size);
  virtual ~Electron_Pair_List() = default;
  virtual void move(Random&, const Electron_Pair_GTO_Weight&) = 0;
  virtual bool requires_blocking() = 0;

  // functions to emulate vector interface
  std::size_t size() const {
    return electron_pairs.size();
  }

  std::vector<std::array<double, 3>> pos1;
  std::vector<std::array<double, 3>> pos2;
  std::vector<double> wgt;
  std::vector<double> rv;
  std::vector<double> r12;
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
Electron_Pair_List* create_electron_pair_sampler(IOPs& iops, Molecule& molec, Electron_Pair_GTO_Weight& weight);
#endif  // ELECTRON_PAIR_LIST_H_

#ifndef METROPOLIS_ELECTRON_PAIR_LIST_H_
#define METROPOLIS_ELECTRON_PAIR_LIST_H_

#include "electron_pair_list.h"

class Metropolis_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Metropolis_Electron_Pair_List(int size, double ml, Random& random, const Molec& molec, const Electron_Pair_GTO_Weight& weight);
  ~Metropolis_Electron_Pair_List() override = default;
  void move(Random& random, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void initialize(Electron_Pair&, Random&, const Molec&, const Electron_Pair_GTO_Weight&);
  void mc_move_scheme(Electron_Pair&, Random&, const Electron_Pair_GTO_Weight&);
  void rescale_move_length();

  double move_length;
  int moves_since_rescale;
  int successful_moves;
  int failed_moves;
};
#endif  // METROPOLIS_ELECTRON_PAIR_LIST_H_

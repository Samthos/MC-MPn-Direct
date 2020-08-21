#ifndef DIRECT_ELECTRON_PAIR_LIST_H_
#define DIRECT_ELECTRON_PAIR_LIST_H_

#include "electron_pair_list.h"

class Dummy_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Dummy_Electron_Pair_List(int size) : Electron_Pair_List(size) {
    std::fill(wgt.begin(), wgt.end(), 1.0);
    std::fill(rv.begin(), rv.end(), 1.0);
  }
  ~Dummy_Electron_Pair_List() override = default;
  void move(Random& random, const Electron_Pair_GTO_Weight& weight) override {};
  bool requires_blocking() override {return false;};

 private:
};
#endif  // DIRECT_ELECTRON_PAIR_LIST_H_

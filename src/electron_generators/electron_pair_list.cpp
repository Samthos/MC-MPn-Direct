#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "electron_pair_list.h"

std::ostream& operator << (std::ostream& os, const Electron_Pair& electron_pair) {
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.wgt << ",";
  return os;
}

Electron_Pair_List::Electron_Pair_List(int size) :
    electron_pairs(size),
    pos1(size),
    pos2(size),
    wgt(size),
    rv(size),
    r12(size) {}
double Electron_Pair_List::calculate_r12(const Electron_Pair &electron_pair_list) {
  double r12;
  std::array<double, 3> dr{};
  std::transform(electron_pair_list.pos1.begin(), electron_pair_list.pos1.end(), electron_pair_list.pos2.begin(), dr.begin(),
                 std::minus<>());
  r12 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
  return sqrt(r12);
}
void Electron_Pair_List::set_weight(Electron_Pair& electron_pair, const Electron_Pair_GTO_Weight& weight) {
  electron_pair.wgt = weight.weight(electron_pair.pos1, electron_pair.pos2);
  electron_pair.r12 = calculate_r12(electron_pair);
  electron_pair.rv = 1.0 / (electron_pair.r12 * electron_pair.wgt);
}
void Electron_Pair_List::transpose() {
  for (size_t i = 0; i < electron_pairs.size(); i++) {
    pos1[i] = electron_pairs[i].pos1;
    pos2[i] = electron_pairs[i].pos2;
    wgt[i] = electron_pairs[i].wgt;
    rv[i] = electron_pairs[i].rv;
    r12[i] = electron_pairs[i].r12;
  }
}


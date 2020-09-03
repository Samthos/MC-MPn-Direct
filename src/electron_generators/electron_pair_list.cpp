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

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Electron_Pair_List<Container, Allocator>::Electron_Pair_List(int size) :
    electron_pairs(size),
    pos1(size),
    pos2(size),
    wgt(size),
    inverse_weight(size),
    rv(size),
    r12(size) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Electron_Pair_List<Container, Allocator>::set_weight(Electron_Pair& electron_pair, const Electron_Pair_GTO_Weight& weight) {
  electron_pair.wgt = weight.weight(electron_pair.pos1, electron_pair.pos2);
  electron_pair.r12 = Point::distance(electron_pair.pos1, electron_pair.pos2);
  electron_pair.rv = 1.0 / (electron_pair.r12 * electron_pair.wgt);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Electron_Pair_List<Container, Allocator>::transpose() {
  for (size_t i = 0; i < electron_pairs.size(); i++) {
    pos1[i] = electron_pairs[i].pos1;
    pos2[i] = electron_pairs[i].pos2;
    wgt[i] = electron_pairs[i].wgt;
    inverse_weight[i] = 1.0 / electron_pairs[i].wgt;
    rv[i] = electron_pairs[i].rv;
    r12[i] = electron_pairs[i].r12;
  }
}


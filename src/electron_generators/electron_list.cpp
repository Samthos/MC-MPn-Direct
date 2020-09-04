#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "electron_list.h"

std::ostream& operator << (std::ostream& os, const Electron& electron) {
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.weight << ",";
  return os;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Electron_List<Container, Allocator>::Electron_List(int size) :
    electrons(size),
    pos(size),
    weight(size),
    inverse_weight(size) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Electron_List<Container, Allocator>::set_weight(Electron& electron, const Electron_GTO_Weight& weight) {
  electron.weight = weight.weight(electron.pos);
  electron.inverse_weight = 1.0 / electron.weight;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Electron_List<Container, Allocator>::transpose() {
  for (size_t i = 0; i < electrons.size(); i++) {
    pos[i] = electrons[i].pos;
    weight[i] = electrons[i].weight;
    inverse_weight[i] = electrons[i].inverse_weight;
  }
}


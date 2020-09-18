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
    rv_inverse_weight(size * 2),
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

#ifdef HAVE_CUDA
template <>
Electron_Pair_List<thrust::device_vector, thrust::device_allocator>::Electron_Pair_List(int size) :
    electron_pairs(size),
    pos1(size),
    pos2(size),
    wgt(size),
    inverse_weight(size),
    rv(size),
    rv_inverse_weight(size * 2),
    r12(size),
    m_pos1(size),
    m_pos2(size),
    m_wgt(size),
    m_inverse_weight(size),
    m_rv(size),
    m_r12(size) {}

template <>
void Electron_Pair_List<thrust::device_vector, thrust::device_allocator>::transpose() {
  for (size_t i = 0; i < electron_pairs.size(); i++) {
    m_pos1[i] = electron_pairs[i].pos1;
    m_pos2[i] = electron_pairs[i].pos2;
    m_wgt[i] = electron_pairs[i].wgt;
    m_inverse_weight[i] = 1.0 / electron_pairs[i].wgt;
    m_rv[i] = electron_pairs[i].rv;
    m_r12[i] = electron_pairs[i].r12;
  }
  thrust::copy(m_pos1.begin(), m_pos1.end(), pos1.begin());
  thrust::copy(m_pos2.begin(), m_pos2.end(), pos2.begin());
  thrust::copy(m_wgt.begin(), m_wgt.end(), wgt.begin());
  thrust::copy(m_inverse_weight.begin(), m_inverse_weight.end(), inverse_weight.begin());
  thrust::copy(m_rv.begin(), m_rv.end(), rv.begin());
  thrust::copy(m_r12.begin(), m_r12.end(), r12.begin());


  thrust::copy(rv.begin(), rv.end(), rv_inverse_weight.begin());
  thrust::copy(inverse_weight.begin(), inverse_weight.end(), rv_inverse_weight.begin() + inverse_weight.size());
}
#endif

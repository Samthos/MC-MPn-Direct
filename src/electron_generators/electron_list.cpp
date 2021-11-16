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
    inverse_weight(size)
{}

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

#ifdef HAVE_CUDA
template <>
Electron_List<thrust::device_vector, thrust::device_allocator>::Electron_List(int size) :
    electrons(size),
    pos(size),
    weight(size),
    inverse_weight(size),
    m_pos(size),
    m_weight(size),
    m_inverse_weight(size)
{}

template <>
void Electron_List<thrust::device_vector, thrust::device_allocator>::transpose() {
  for (size_t i = 0; i < electrons.size(); i++) {
    m_pos[i] = electrons[i].pos;
    m_weight[i] = electrons[i].weight;
    m_inverse_weight[i] = electrons[i].inverse_weight;
  }
  thrust::copy(m_pos.begin(), m_pos.end(), pos.begin());
  thrust::copy(m_weight.begin(), m_weight.end(), weight.begin());
  thrust::copy(m_inverse_weight.begin(), m_inverse_weight.end(), inverse_weight.begin());
}
#endif

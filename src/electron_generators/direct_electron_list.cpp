#include "direct_electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Direct_Electron_List<Container, Allocator>::Direct_Electron_List(int size) : Electron_List<Container, Allocator>(size) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Direct_Electron_List<Container, Allocator>::requires_blocking() {
  return false;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Direct_Electron_List<Container, Allocator>::move(Random& random, const Electron_GTO_Weight& weight) {
  for (Electron &electron : this->electrons) {
    mc_move_scheme(electron, random, weight);
  }
  this->transpose();
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Direct_Electron_List<Container, Allocator>::mc_move_scheme(Electron& electron, Random& random, const Electron_GTO_Weight& weight) {
  // choose function to sample;
  double rnd = random.uniform();
  auto it = std::lower_bound(std::begin(weight.cum_sum), std::end(weight.cum_sum), rnd);
  auto index = static_cast<int>(std::distance(weight.cum_sum.begin(), it));
  auto atom = weight.cum_sum_index[index][0];
  auto prim = weight.cum_sum_index[index][1];

  // compute some parameters
  double alpha = 1.0 / sqrt(2.0 * weight.mcBasisList[atom].alpha[prim]);

  // sample x, y, and theta
  for (int i = 0; i < 3; i++) {
    electron.pos[i] = random.normal(weight.mcBasisList[atom].center[i], alpha);
  }

  Electron_List<Container, Allocator>::set_weight(electron, weight);
}


#include "dummy_electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Dummy_Electron_Pair_List<Container, Allocator>::Dummy_Electron_Pair_List(int size) : Electron_Pair_List<Container, Allocator>(size) {
#ifdef HAVE_CUDA
  namespace NS = thrust;
#else
  namespace NS = std;
#endif
  NS::fill(this->wgt.begin(), this->wgt.end(), 1.0);
  NS::fill(this->inverse_weight.begin(), this->inverse_weight.end(), 1.0);
  NS::fill(this->rv.begin(), this->rv.end(), 1.0);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Dummy_Electron_Pair_List<Container, Allocator>::move(Random& random, const Electron_Pair_GTO_Weight& weight) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Dummy_Electron_Pair_List<Container, Allocator>::requires_blocking() {
  return false;
}

#include "dummy_electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Dummy_Electron_List<Container, Allocator>::Dummy_Electron_List(int size) : Electron_List<Container, Allocator>(size) {
#ifdef HAVE_CUDA
  namespace NS = thrust;
#else
  namespace NS = std;
#endif
  std::vector<Point> p;
  for (int i = 0; i < size; i++) {
    p.emplace_back(i, i, i);
  }
  NS::copy(p.begin(), p.end(), this->pos.begin());
  NS::fill(this->weight.begin(), this->weight.end(), 1.0);
  NS::fill(this->inverse_weight.begin(), this->inverse_weight.end(), 1.0);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Dummy_Electron_List<Container, Allocator>::move(Random& random, const Electron_GTO_Weight& weight) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Dummy_Electron_List<Container, Allocator>::requires_blocking() {
  return false;
}

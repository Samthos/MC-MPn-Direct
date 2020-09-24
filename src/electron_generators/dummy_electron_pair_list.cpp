#include "dummy_electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Dummy_Electron_Pair_List<Container, Allocator>::Dummy_Electron_Pair_List(int size) : Electron_Pair_List<Container, Allocator>(size) {
#ifdef HAVE_CUDA
  namespace NS = thrust;
#else
  namespace NS = std;
#endif
  std::vector<Point> p1;
  std::vector<Point> p2;
  std::vector<double> r;
  for (int i = 0; i < size; i++) {
    p1.emplace_back( i,  i,  i);
    p2.emplace_back(-i, -i, -i);
    r.emplace_back(Point::distance(p1[i], p2[i]));
  }
  NS::copy(p1.begin(), p1.end(), this->pos1.begin());
  NS::copy(p2.begin(), p2.end(), this->pos2.begin());
  NS::copy(r.begin(), r.end(), this->r12.begin());

  NS::fill(this->wgt.begin(), this->wgt.end(), 1.0);
  NS::fill(this->inverse_weight.begin(), this->inverse_weight.end(), 1.0);
  NS::fill(this->rv.begin(), this->rv.end(), 1.0);

  NS::copy(this->rv.begin(), this->rv.end(), this->rv_inverse_weight.begin());
  NS::copy(this->inverse_weight.begin(), this->inverse_weight.end(), this->rv_inverse_weight.begin() + size);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Dummy_Electron_Pair_List<Container, Allocator>::move(Random& random, const Electron_Pair_GTO_Weight& weight) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Dummy_Electron_Pair_List<Container, Allocator>::requires_blocking() {
  return false;
}

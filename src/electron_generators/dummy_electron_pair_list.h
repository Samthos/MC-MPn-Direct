#ifndef DUMMY_ELECTRON_PAIR_LIST_H_
#define DUMMY_ELECTRON_PAIR_LIST_H_

#include "electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Dummy_Electron_Pair_List : public Electron_Pair_List<Container, Allocator> {
 public:
  explicit Dummy_Electron_Pair_List(int size);
  void move(Random& random, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
};

template class Dummy_Electron_Pair_List<std::vector, std::allocator>;
typedef Dummy_Electron_Pair_List<std::vector, std::allocator> Dummy_Electron_Pair_List_Host;

#ifdef HAVE_CUDA
template class Dummy_Electron_Pair_List<thrust::device_vector, thrust::device_allocator>;
typedef Dummy_Electron_Pair_List<thrust::device_vector, thrust::device_allocator> Dummy_Electron_Pair_List_Device;
#endif
#endif  // DUMMY_ELECTRON_PAIR_LIST_H_

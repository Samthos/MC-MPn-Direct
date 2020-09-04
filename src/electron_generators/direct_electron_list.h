#ifndef DIRECT_ELECTRON_LIST_H_
#define DIRECT_ELECTRON_LIST_H_

#include "electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Direct_Electron_List : public Electron_List<Container, Allocator> {
 public:
  explicit Direct_Electron_List(int size);
  ~Direct_Electron_List() override = default;
  void move(Random& random, const Electron_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void mc_move_scheme(Electron&, Random&, const Electron_GTO_Weight&);
};

template class Direct_Electron_List<std::vector, std::allocator>;
typedef Direct_Electron_List<std::vector, std::allocator> Direct_Electron_List_Host;

#ifdef HAVE_CUDA
template class Direct_Electron_List<thrust::device_vector, thrust::device_allocator>;
typedef Direct_Electron_List<thrust::device_vector, thrust::device_allocator> Direct_Electron_List_Device;
#endif
#endif  // DIRECT_ELECTRON_LIST_H_

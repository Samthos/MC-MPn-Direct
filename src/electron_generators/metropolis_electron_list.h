#ifndef METROPOLIS_ELECTRON_LIST_H_
#define METROPOLIS_ELECTRON_LIST_H_

#include "electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Metropolis_Electron_List : public Electron_List<Container, Allocator> {
 public:
  explicit Metropolis_Electron_List(int size, double ml, Random& random, const Molecule& molec, const Electron_GTO_Weight& weight);
  ~Metropolis_Electron_List() override = default;
  void move(Random& random, const Electron_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void initialize(Electron&, Random&, const Molecule&, const Electron_GTO_Weight&);
  void mc_move_scheme(Electron&, Random&, const Electron_GTO_Weight&);
  void rescale_move_length();

  double move_length;
  int moves_since_rescale;
  int successful_moves;
  int failed_moves;
};

template class Metropolis_Electron_List<std::vector, std::allocator>;
typedef Metropolis_Electron_List<std::vector, std::allocator> Metropolis_Electron_List_Host;

#ifdef HAVE_CUDA
template class Metropolis_Electron_List<thrust::device_vector, thrust::device_allocator>;
typedef Metropolis_Electron_List<thrust::device_vector, thrust::device_allocator> Metropolis_Electron_List_Device;
#endif
#endif  // METROPOLIS_ELECTRON_LIST_H_

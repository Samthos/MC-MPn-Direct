#ifndef DIRECT_ELECTRON_PAIR_LIST_H_
#define DIRECT_ELECTRON_PAIR_LIST_H_

#include "electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Direct_Electron_Pair_List : public Electron_Pair_List<Container, Allocator> {
 public:
  explicit Direct_Electron_Pair_List(int size);
  void move(Random& random, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void mc_move_scheme(Electron_Pair&, Random&, const Electron_Pair_GTO_Weight&);
  static double calculate_r(Random& random, double alpha, double beta, double a);
  static double calculate_phi(double p, double r, double alpha, double beta, double a);
  static double CDF(const double& rho, const double& c, const double& erf_c);
  static double PDF(const double& rho, const double& c, const double& erf_c);
  static double PDF_Prime(const double& rho, const double& c, const double& erf_c);
};

template class Direct_Electron_Pair_List<std::vector, std::allocator>;
typedef Direct_Electron_Pair_List<std::vector, std::allocator> Direct_Electron_Pair_List_Host;

#ifdef HAVE_CUDA
template class Direct_Electron_Pair_List<thrust::device_vector, thrust::device_allocator>;
typedef Direct_Electron_Pair_List<thrust::device_vector, thrust::device_allocator> Direct_Electron_Pair_List_Device;
#endif
#endif  // DIRECT_ELECTRON_PAIR_LIST_H_

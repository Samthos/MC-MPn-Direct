#ifndef ELECTRON_PAIR_LIST_H_
#define ELECTRON_PAIR_LIST_H_
#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#endif

#include <vector>

#include "samplers.h"
#include "weight_function.h"
#include "molecule.h"
#include "../qc_random.h"

struct Electron_Pair {
  Point pos1, pos2;
  double wgt;
  double rv;
  double r12;
};
std::ostream& operator << (std::ostream& os, const Electron_Pair& electron_pair);

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Electron_Pair_List {
  typedef Container<double, Allocator<double>> vector_double;
  typedef Container<Point, Allocator<Point>> vector_Point;

 public:
  explicit Electron_Pair_List(int size);
  virtual ~Electron_Pair_List() = default;
  virtual void move(Random&, const Electron_Pair_GTO_Weight&) = 0;
  virtual bool requires_blocking() = 0;

  // functions to emulate vector interface
  std::size_t size() const {
    return electron_pairs.size();
  }

  vector_Point pos1;
  vector_Point pos2;
  vector_double wgt;
  vector_double inverse_weight;
  vector_double rv;
  vector_double r12;

 protected:
  static double calculate_r12(const Electron_Pair &electron_pair);
  static void set_weight(Electron_Pair&, const Electron_Pair_GTO_Weight&);
  void transpose();
  std::vector<Electron_Pair>::iterator begin() {
    return electron_pairs.begin();
  }
  std::vector<Electron_Pair>::iterator end() {
    return electron_pairs.end();
  }

  std::vector<Point> m_pos1;
  std::vector<Point> m_pos2;
  std::vector<double> m_wgt;
  std::vector<double> m_inverse_weight;
  std::vector<double> m_rv;
  std::vector<double> m_r12;
  std::vector<Electron_Pair> electron_pairs;
};

template class Electron_Pair_List<std::vector, std::allocator>;
typedef Electron_Pair_List<std::vector, std::allocator> Electron_Pair_List_Host;

#ifdef HAVE_CUDA
template <> Electron_Pair_List<thrust::device_vector, thrust::device_allocator>::Electron_Pair_List(int size);
template <> void Electron_Pair_List<thrust::device_vector, thrust::device_allocator>::transpose();
template class Electron_Pair_List<thrust::device_vector, thrust::device_allocator>;
typedef Electron_Pair_List<thrust::device_vector, thrust::device_allocator> Electron_Pair_List_Device;
#endif
#endif  // ELECTRON_PAIR_LIST_H_

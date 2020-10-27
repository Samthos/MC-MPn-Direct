#ifndef ELECTRON_LIST_H_
#define ELECTRON_LIST_H_
#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#endif

#include <vector>

#include "samplers.h"
#include "weight_function.h"
#include "molecule.h"
#include "../qc_random.h"

struct Electron {
  Point pos;
  double weight;
  double inverse_weight;
};
std::ostream& operator << (std::ostream& os, const Electron& electron);

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Electron_List {
  typedef Container<double, Allocator<double>> vector_double;
  typedef Container<Point, Allocator<Point>> vector_Point;

 public:
  explicit Electron_List(int size);
  virtual ~Electron_List() = default;
  virtual void move(Random&, const Electron_GTO_Weight&) = 0;
  virtual bool requires_blocking() = 0;

  // functions to emulate vector interface
  std::size_t size() const {
    return electrons.size();
  }

  vector_Point pos;
  vector_double weight;
  vector_double inverse_weight;

 protected:
  static void set_weight(Electron&, const Electron_GTO_Weight&);
  void transpose();
  std::vector<Electron>::iterator begin() {
    return electrons.begin();
  }
  std::vector<Electron>::iterator end() {
    return electrons.end();
  }

  std::vector<Point> m_pos;
  std::vector<double> m_weight;
  std::vector<double> m_inverse_weight;
  std::vector<Electron> electrons;
};

template class Electron_List<std::vector, std::allocator>;
typedef Electron_List<std::vector, std::allocator> Electron_List_Host;

#ifdef HAVE_CUDA
template <> Electron_List<thrust::device_vector, thrust::device_allocator>::Electron_List(int size);
template <> void Electron_List<thrust::device_vector, thrust::device_allocator>::transpose();
template class Electron_List<thrust::device_vector, thrust::device_allocator>;
typedef Electron_List<thrust::device_vector, thrust::device_allocator> Electron_List_Device;
#endif
#endif  // ELECTRON_LIST_H_

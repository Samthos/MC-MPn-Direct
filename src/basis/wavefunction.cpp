#include <vector>
#include "wavefunction.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Wavefunction<Container, Allocator>::Wavefunction(std::vector<Point>* p, const std::shared_ptr<Movec_Parser> movecs_in) :
  iocc1(movecs_in->iocc1),
  iocc2(movecs_in->iocc2),
  ivir1(movecs_in->ivir1),
  ivir2(movecs_in->ivir2),
  movecs(movecs_in->movecs),
  electrons(p->size()),
  lda(ivir2),
  psi(lda * electrons, 0.0),
  psiTau(lda * electrons, 0.0),
  pos(p)
{ }

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::data() const {
  return get_raw_pointer(psi);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::occ() const {
  return get_raw_pointer(psi) + iocc1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::vir() const {
  return get_raw_pointer(psi) + ivir1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::dataTau() const {
  return get_raw_pointer(psiTau);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::occTau() const {
  return get_raw_pointer(psiTau) + iocc1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::virTau() const {
  return get_raw_pointer(psiTau) + ivir1;
}


template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::data() {
  return get_raw_pointer(psi);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::occ() {
  return get_raw_pointer(psi) + iocc1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::vir() {
  return get_raw_pointer(psi) + ivir1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::dataTau() {
  return get_raw_pointer(psiTau);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::occTau() {
  return get_raw_pointer(psiTau) + iocc1;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::virTau() {
  return get_raw_pointer(psiTau) + ivir1;
}


template <template <typename, typename> typename Container, template <typename> typename Allocator>
double* Wavefunction<Container, Allocator>::get_raw_pointer(Container<double, Allocator<double>>&) {
  return nullptr;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
const double* Wavefunction<Container, Allocator>::get_raw_pointer(const Container<double, Allocator<double>>&) {
  return nullptr;
}


template <>
double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(std::vector<double, std::allocator<double>>& v) {
  return v.data();
}

template <>
const double* Wavefunction<std::vector, std::allocator>::get_raw_pointer(const std::vector<double, std::allocator<double>>& v) {
  return v.data();
}

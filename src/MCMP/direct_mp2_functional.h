#ifndef DIRECT_MP2_Functional_H_
#define DIRECT_MP2_Functional_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "mp_functional.h"

template <int CVMP2>
class Direct_MP2_Functional : public Direct_MP_Functional<std::vector, std::allocator> {
 public:
  Direct_MP2_Functional() : Direct_MP_Functional(CVMP2 * (CVMP2+1), 1, "22") {}
  void energy(double& emp, std::vector<double>& control, Wavefunction_Type&, Wavefunction_Type&, Electron_Pair_List*, Tau*) override;
 private:
};

template class Direct_MP2_Functional<0>;
template class Direct_MP2_Functional<1>;
template class Direct_MP2_Functional<2>;

#endif  // MP2_Functional_H_

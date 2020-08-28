#ifndef MP2_Functional_H_
#define MP2_Functional_H_

#include "mp_functional.h"

template <int CVMP2> 
class MP2_Functional : public Standard_MP_Functional<std::vector, std::allocator> {
 public:
   MP2_Functional() : Standard_MP_Functional<std::vector, std::allocator>(CVMP2 * (CVMP2+1), 1, "22") {}
   void energy(double& emp, std::vector<double>& control, OVPS_Type&, Electron_Pair_List*, Tau*) override;
};

template class MP2_Functional<0>;
template class MP2_Functional<1>;
template class MP2_Functional<2>;
#endif  // MP2_Functional_H_

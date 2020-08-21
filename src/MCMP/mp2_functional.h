#ifndef MP2_Functional_H_
#define MP2_Functional_H_

#include "mp_functional.h"

template <int CVMP2>
class MP2_Functional : public Standard_MP_Functional {
 public:
   MP2_Functional() : Standard_MP_Functional(CVMP2 * (CVMP2+1), 1, "22") {}
   void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List*, Tau*) override;
 private:
};

template class MP2_Functional<0>;
template class MP2_Functional<1>;
template class MP2_Functional<2>;

MP_Functional* create_MP2_Functional(int cv_level);

template <int CVMP2>
class Fast_MP2_Functional : public Fast_MP_Functional {
 public:
  Fast_MP2_Functional() : Fast_MP_Functional(CVMP2 * (CVMP2+1), 1, "22") {}
  void energy(double& emp, std::vector<double>& control, Wavefunction_Type&, Wavefunction_Type&, Electron_Pair_List*, Tau*) override;
 private:
};

template class Fast_MP2_Functional<0>;
template class Fast_MP2_Functional<1>;
template class Fast_MP2_Functional<2>;

MP_Functional* create_Fast_MP2_Functional(int cv_level);
#endif  // MP2_Functional_H_

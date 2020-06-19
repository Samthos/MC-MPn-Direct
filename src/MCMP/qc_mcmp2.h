#ifndef QC_MCMP2_H_
#define QC_MCMP2_H_

#include "mcmp.h"

template <int CVMP2>
class MCMP2 : public MCMP {
 public:
   MCMP2() : MCMP(CVMP2 * (CVMP2+1), 1, "22", false) {}
   void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List*, Tau*) override;
  void energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list){}
 private:
};

template class MCMP2<0>;
template class MCMP2<1>;
template class MCMP2<2>;

MCMP* create_MCMP2(int cv_level);
#endif  // QC_MCMP2_H_

#ifndef QC_MCMP2_H_
#define QC_MCMP2_H_

#include "mcmp.h"

template <int CVMP2>
class MCMP2 : public MCMP {
 public:
   MCMP2() : MCMP(6, 1) {}
   void energy(double& emp, std::vector<double>& control, OVPs&, Electron_Pair_List*, Tau*) override;
 private:
};

template class MCMP2<0>;
template class MCMP2<1>;
template class MCMP2<2>;
#endif  // QC_MCMP2_H_

#ifndef QC_MCMP3_H_
#define QC_MCMP3_H_

#include "mcmp.h"

template <int CVMP3>
class MCMP3 : public MCMP {
 public:
  MCMP3() : MCMP(36, 2, "23") {}
  void energy(double& emp, std::vector<double>& control, OVPs&, Electron_Pair_List*, Tau*) override;
 private:
  void mcmp3_helper(
    double& en3, std::vector<double>& control, const int offset,
    unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv, std::vector<double>& wgt);
};

template class MCMP3<0>;
template class MCMP3<1>;
template class MCMP3<2>;
template class MCMP3<3>;

MCMP* create_MCMP3(int cv_level);
#endif  // QC_MCMP3_H_

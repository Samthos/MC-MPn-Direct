#ifndef MP3_Functional_H_
#define MP3_Functional_H_

#include "mp_functional.h"

template <int CVMP3>
class MP3_Functional : public Standard_MP_Functional<std::vector, std::allocator> {
 public:
  MP3_Functional() : Standard_MP_Functional<std::vector, std::allocator>(3 * CVMP3 * (1 + CVMP3), 2, "23") {}
  void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List*, Tau*) override;
 private:
  void mcmp3_helper(
    double& en3, std::vector<double>& control, const int offset,
    unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv, std::vector<double>& wgt);
};

template class MP3_Functional<0>;
template class MP3_Functional<1>;
template class MP3_Functional<2>;
template class MP3_Functional<3>;

MP_Functional* create_MP3_Functional(int cv_level);
#endif  // MP3_Functional_H_

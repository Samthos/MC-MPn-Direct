#ifndef MP3_Functional_H_
#define MP3_Functional_H_

#include "mp_functional.h"

template <int CVMP3>
class MP3_Functional : public Standard_MP_Functional<std::vector, std::allocator> {
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
 public:
  explicit MP3_Functional(int);
  void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List_Type*, Tau*) override;

 private:
  void mcmp3_helper(
    double& en3, std::vector<double>& control, const int offset,
    unsigned int electron_pairs, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv, std::vector<double>& wgt);

  vector_double A_ij;
  vector_double A_ik;
  vector_double A_jk;
};

template class MP3_Functional<0>;
template class MP3_Functional<1>;
template class MP3_Functional<2>;
template class MP3_Functional<3>;

MP_Functional* create_MP3_Functional(int cv_level, int electron_pairs);
#endif  // MP3_Functional_H_

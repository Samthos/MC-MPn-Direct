#ifndef OVPS_SET_H_
#define OVPS_SET_H_

#include <vector>

// class OVPS_SET_DEVICE;
class OVPS_SET {
 public:
  OVPS_SET() = default;
  OVPS_SET(int mc_pair_num_);
  void resize(int mc_pair_num_);
  void update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda);

  int mc_pair_num;
  std::vector<double> s_11, s_12, s_21, s_22;
  // OVPS_SET& operator = (const OVPS_SET_DEVICE&);
};
#endif  // OVPS_SET_H_

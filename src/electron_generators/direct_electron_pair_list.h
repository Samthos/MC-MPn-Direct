#ifndef DIRECT_ELECTRON_PAIR_LIST_H_
#define DIRECT_ELECTRON_PAIR_LIST_H_

#include "electron_pair_list.h"

class Direct_Electron_Pair_List : public Electron_Pair_List {
 public:
  explicit Direct_Electron_Pair_List(int size) : Electron_Pair_List(size) {}
  ~Direct_Electron_Pair_List() override = default;
  void move(Random& random, const Electron_Pair_GTO_Weight& weight) override;
  bool requires_blocking() override;

 private:
  static void mc_move_scheme(Electron_Pair&, Random&, const Electron_Pair_GTO_Weight&);
  static double calculate_r(Random& random, double alpha, double beta, double a);
  static double calculate_phi(double p, double r, double alpha, double beta, double a);
  static double CDF(const double& rho, const double& c, const double& erf_c);
  static double PDF(const double& rho, const double& c, const double& erf_c);
  static double PDF_Prime(const double& rho, const double& c, const double& erf_c);
};
#endif  // DIRECT_ELECTRON_PAIR_LIST_H_

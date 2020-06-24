#ifndef MBPT_H_
#define MBPT_H_
#include <vector>
#include <unordered_map>

#include "../qc_ovps.h"
#include "tau.h"
#include "../electron_pair_list.h"
#include "../electron_list.h"

class MCMP {
 public:
  MCMP(int ncv, int ntc, const std::string& e, bool f) : n_control_variates(ncv), n_tau_coordinates(ntc), extension(e), is_f12(f) {}
  virtual void energy(double& emp, std::vector<double>& control, OVPS_Host&, Electron_Pair_List*, Tau*) = 0;
  virtual void energy_f12(double& emp, std::vector<double>& control, std::unordered_map<int, Wavefunction>& wavefunctions, const Electron_Pair_List* electron_pair_list, const Electron_List* electron_list) = 0;

  int n_control_variates;
  int n_tau_coordinates;
  std::string extension;
  bool is_f12;

 protected:
};

#endif  // MBPT_H_

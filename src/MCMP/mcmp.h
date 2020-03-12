#ifndef MBPT_H_
#define MBPT_H_
#include <vector>

#include "../qc_ovps.h"
#include "../tau_integrals.h"
#include "../electron_pair_list.h"

class MCMP {
 public:
   MCMP(int n, int m) : n_control_variates(n), n_tau_coordinates(m) {}
   virtual void energy(double& emp, std::vector<double>& control, OVPs&, Electron_Pair_List*, Tau*) = 0;

   int n_control_variates;
   int n_tau_coordinates;

 protected:
};

#endif  // MBPT_H_

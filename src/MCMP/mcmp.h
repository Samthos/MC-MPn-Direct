#ifndef MBPT_H_
#define MBPT_H_
#include <vector>

#include "../qc_ovps.h"
#include "../tau_integrals.h"
#include "../electron_pair_list.h"

class MCMP {
 public:
   MCMP(int ncv, int ntc, const std::string& e) : n_control_variates(ncv), n_tau_coordinates(ntc), extension(e) {}
   virtual void energy(double& emp, std::vector<double>& control, OVPs&, Electron_Pair_List*, Tau*) = 0;

   int n_control_variates;
   int n_tau_coordinates;
   std::string extension;

 protected:
};

#endif  // MBPT_H_

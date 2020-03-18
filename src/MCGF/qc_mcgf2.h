#ifndef QC_MCGF2_H_
#define QC_MCGF2_H_

#include <unordered_map>
#include <vector>

#include "qc_mcgf.h"
#include "../qc_input.h"
#include "../basis/qc_basis.h"
#include "../tau_integrals.h"
#include "../electron_pair_list.h"

class GF2_Functional : public MCGF {
  public:
   GF2_Functional(IOPs&, Basis&);
   void energy_f12(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Electron_List*);

  private:
   void core(OVPs& ovps, Electron_Pair_List* electron_pair_list);
   void energy_no_diff(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*);
   void energy_diff(std::vector<std::vector<double>>&,
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*);

   std::vector<double> en2mCore;
   std::vector<double> en2pCore;
};
#endif  // QC_MCGF2_H_

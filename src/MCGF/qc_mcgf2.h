#ifndef QC_MCGF2_H_
#define QC_MCGF2_H_

#include <unordered_map>
#include <vector>

#include "../qc_input.h"
#include "../basis/qc_basis.h"
#include "../tau_integrals.h"
#include "../electron_pair_list.h"

class GF2_Functional {
  public:
   GF2_Functional(IOPs&, Basis&);
   void energy(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       OVPs&, Electron_Pair_List*, Tau*);
   void energy_full();
   void energy_full_diff();

  private:
   void core(OVPs& ovps, Electron_Pair_List* electron_pair_list);
   void energy_no_diff(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*);
   void energy_diff(std::vector<std::vector<double>>&,
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*);

   int n_electron_pairs, numBand, offBand, numDiff;
   double nsamp;

   std::vector<double> en2mCore;
   std::vector<double> en2pCore;
   std::vector<double> enCore;
   std::vector<double> ent;
};
#endif  // QC_MCGF2_H_

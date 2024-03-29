#ifndef QC_MCGF2_H_
#define QC_MCGF2_H_

#include <unordered_map>
#include <vector>

#include "qc_mcgf.h"
#include "../qc_input.h"
#include "../basis/basis.h"
#include "electron_pair_list.h"
#include "tau.h"

class GF2_Functional : public MCGF {
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List_Host Electron_List_Type;
  public:
   GF2_Functional(IOPs&);
   void energy_f12(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction_Type>&,
       Electron_Pair_List_Type*, Electron_List_Type*);

  private:
   void core(OVPS_Host& ovps, Electron_Pair_List_Type* electron_pair_list);
   void energy_no_diff(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction_Type>&,
       Electron_Pair_List_Type*, Tau*);
   void energy_diff(std::vector<std::vector<double>>&,
       std::unordered_map<int, Wavefunction_Type>&,
       Electron_Pair_List_Type*, Tau*);

   std::vector<double> en2mCore;
   std::vector<double> en2pCore;
};
#endif  // QC_MCGF2_H_

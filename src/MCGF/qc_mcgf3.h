#ifndef QC_MCGF3_H_
#define QC_MCGF3_H_

#include <unordered_map>
#include <vector>

#include "qc_mcgf.h"
#include "../qc_input.h"
#include "../basis/basis.h"
#include "tau.h"
#include "electron_pair_list.h"

class GF3_Functional : public MCGF {
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List_Host Electron_List_Type;
  public:
   GF3_Functional(IOPs&);
   ~GF3_Functional();
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
   void gf3_core_c(OVPS_Host&, Electron_Pair_List_Type*);
   void gf3_core_1(OVPS_Host&, Electron_Pair_List_Type*);
   void gf3_core_2(OVPS_Host&, Electron_Pair_List_Type*);
   void gf3_core_12(OVPS_Host&, Electron_Pair_List_Type*);

   double* rv;
   std::array<std::vector<double>, 3> T;
   std::vector<double> one;
   std::vector<double> en3c12;
   std::vector<double> en3c22;
   std::vector<double> en3_1pCore ;
   std::vector<double> en3_2pCore ;
   std::vector<double> en3_12pCore;
   std::vector<double> en3_1mCore ;
   std::vector<double> en3_2mCore ;
   std::vector<double> en3_12mCore;
   std::vector<double> en3_12cCore;
   std::vector<double> en3_22cCore;
};
#endif  // QC_MCGF3_H_

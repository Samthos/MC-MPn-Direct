#ifndef MCGF_H_
#define MCGF_H_

#include <unordered_map>
#include <vector>

#include "../qc_input.h"
#include "tau.h"
#include "electron_pair_list.h"
#include "electron_list.h"
#include "../qc_ovps.h"

class MCGF {
  public:
   MCGF(IOPs&, int ntc, std::string ext, bool f);
   void energy(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       OVPS_Host&, Electron_Pair_List*, Tau*);
   virtual void energy_f12(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Electron_List*) = 0;

   int n_tau_coordinates;
   std::string extension;
   bool is_f12;

  protected:
   virtual void core(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) = 0;
   virtual void energy_no_diff(std::vector<std::vector<double>>&, 
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*) = 0;
   virtual void energy_diff(std::vector<std::vector<double>>&,
       std::unordered_map<int, Wavefunction>&,
       Electron_Pair_List*, Tau*) = 0;


   int n_electron_pairs, numBand, offBand, numDiff;
   double nsamp;
   std::vector<double> ent;
};

#endif  // MCGF_H_

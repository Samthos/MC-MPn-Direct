#include "electron_pair_list.h"
#include "direct_electron_pair_list.h"
#include "metropolis_electron_pair_list.h"

Electron_Pair_List* create_electron_pair_sampler(IOPs& iops, Molec& molec, Electron_Pair_GTO_Weight& weight) {
  Electron_Pair_List* electron_pair_list = nullptr;
  if (iops.iopns[KEYS::SAMPLER] == SAMPLER::DIRECT) {
    electron_pair_list = new Direct_Electron_Pair_List(iops.iopns[KEYS::ELECTRON_PAIRS]);
  } else if (iops.iopns[KEYS::SAMPLER] == SAMPLER::METROPOLIS) {
    std::string str = iops.sopns[KEYS::SEED_FILE];
    if (!str.empty()) {
      str += ".electron_pair_list_metropolis";
    }
    Random rnd(iops.iopns[KEYS::DEBUG], str);
    electron_pair_list = new Metropolis_Electron_Pair_List(iops.iopns[KEYS::ELECTRON_PAIRS], iops.dopns[KEYS::MC_DELX], rnd, molec, weight);
  }
  return electron_pair_list;
}


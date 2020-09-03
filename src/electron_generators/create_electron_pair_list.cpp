#include "samplers.h"
#include "electron_pair_list.h"
#include "direct_electron_pair_list.h"
#include "metropolis_electron_pair_list.h"

Electron_Pair_List<std::vector, std::allocator>* create_electron_pair_sampler(Molecule& molec,
    Electron_Pair_GTO_Weight& weight,
    int sampler_type,
    size_t electron_pairs,
    double delx,
    int debug,
    std::string seed_file) {
  Electron_Pair_List<std::vector, std::allocator>* electron_pair_list = nullptr;
  if (sampler_type == SAMPLER::DIRECT) {
    electron_pair_list = new Direct_Electron_Pair_List<std::vector, std::allocator>(electron_pairs);
  } else if (sampler_type == SAMPLER::METROPOLIS) {
    if (!seed_file.empty()) {
      seed_file += ".electron_pair_list_metropolis";
    }
    Random rnd(debug, seed_file);
    electron_pair_list = new Metropolis_Electron_Pair_List<std::vector, std::allocator>(electron_pairs, delx, rnd, molec, weight);
  }
  return electron_pair_list;
}


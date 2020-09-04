#ifndef CREATE_ELECTRON_LIST_H_
#define CREATE_ELECTRON_LIST_H_
#include "samplers.h"
#include "electron_list.h"
#include "direct_electron_list.h"
#include "metropolis_electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Electron_List<Container, Allocator>* create_electron_sampler(Molecule& molec,
    Electron_GTO_Weight& weight,
    int sampler_type,
    size_t electrons,
    double delx,
    int debug,
    std::string seed_file) {
  Electron_List<Container, Allocator>* electron_list = nullptr;
  if (sampler_type == SAMPLER::DIRECT) {
    electron_list = new Direct_Electron_List<Container, Allocator>(electrons);
  } else if (sampler_type == SAMPLER::METROPOLIS) {
    if (!seed_file.empty()) {
      seed_file += ".electron_list_metropolis";
    }
    Random rnd(debug, seed_file);
    electron_list = new Metropolis_Electron_List<Container, Allocator>(electrons, delx, rnd, molec, weight);
  }
  return electron_list;
}

#endif  // CREATE_ELECTRON_LIST_H_

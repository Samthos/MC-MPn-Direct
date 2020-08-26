#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <bitset>

#include <unordered_map>

#include "qc_monte.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
QC_monte<Container, Allocator>::QC_monte(MPI_info p0, IOPs p1, Molecule p2, Basis_Host p3) :
    mpi_info(p0),
    iops(p1),
    molec(p2),
    basis(std::max(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]), Basis_Parser(iops.sopns[KEYS::BASIS], iops.bopns[KEYS::SPHERICAL], mpi_info, molec)),
    electron_pair_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    electron_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    random(iops.iopns[KEYS::DEBUG], iops.sopns[KEYS::SEED_FILE])
{
  numBand = iops.iopns[KEYS::NUM_BAND];
  offBand = iops.iopns[KEYS::OFF_BAND];
  nDeriv = iops.iopns[KEYS::DIFFS];

  auto movecs = create_movec_parser(mpi_info, molec, MOVEC_TYPE::NWCHEM_BINARY, iops.sopns[KEYS::MOVECS], iops.bopns[KEYS::FREEZE_CORE]);

  iocc1 = movecs->iocc1;
  iocc2 = movecs->iocc2;
  ivir1 = movecs->ivir1;
  ivir2 = movecs->ivir2;

  electron_pair_list = create_electron_pair_sampler(iops, molec, electron_pair_weight);
  wavefunctions.emplace(WC::electron_pairs_1, Wavefunction_Type(&electron_pair_list->pos1, movecs));
  wavefunctions.emplace(WC::electron_pairs_2, Wavefunction_Type(&electron_pair_list->pos2, movecs));

  electron_list = nullptr;
  if (iops.iopns[KEYS::TASK] & TASK::ANY_F12) {
    electron_list = create_electron_sampler(iops, molec, electron_weight);
    wavefunctions.emplace(WC::electrons, Wavefunction_Type(&electron_list->pos, movecs));
    if (iops.iopns[KEYS::TASK] & TASK::ANY_F12_VBX) {
      wavefunctions.emplace(WC::electrons_dx, Wavefunction_Type(&electron_list->pos, movecs));
      wavefunctions.emplace(WC::electrons_dy, Wavefunction_Type(&electron_list->pos, movecs));
      wavefunctions.emplace(WC::electrons_dz, Wavefunction_Type(&electron_list->pos, movecs));

      wavefunctions.emplace(WC::electron_pairs_1_dx, Wavefunction_Type(&electron_pair_list->pos1, movecs));
      wavefunctions.emplace(WC::electron_pairs_1_dy, Wavefunction_Type(&electron_pair_list->pos1, movecs));
      wavefunctions.emplace(WC::electron_pairs_1_dz, Wavefunction_Type(&electron_pair_list->pos1, movecs));

      wavefunctions.emplace(WC::electron_pairs_2_dx, Wavefunction_Type(&electron_pair_list->pos2, movecs));
      wavefunctions.emplace(WC::electron_pairs_2_dy, Wavefunction_Type(&electron_pair_list->pos2, movecs));
      wavefunctions.emplace(WC::electron_pairs_2_dz, Wavefunction_Type(&electron_pair_list->pos2, movecs));
    }
  }
  for (auto &it : wavefunctions) {
    wavefunction_groups[it.first & WS::mask].push_back(it.first);
  }

  //initialize walkers
  tau = create_tau_sampler(static_cast<TAU_GENERATORS::TAU_GENERATORS>(iops.iopns[KEYS::TAU_GENERATORS]), movecs);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
QC_monte<Container, Allocator>::~QC_monte() {
  delete tau;
  delete electron_pair_list;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void QC_monte<Container, Allocator>::move_walkers() {
  electron_pair_list->move(random, electron_pair_weight);
  if (electron_list != nullptr) {
    electron_list->move(random, electron_weight);
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void QC_monte<Container, Allocator>::update_wavefunction() {
  for (auto &it : wavefunction_groups) {
    if (it.second.size() == 1) { // if group size is one, then no derivatives are required
      basis.build_contractions(*wavefunctions[it.second.front()].pos);
    } else {
      basis.build_contractions_with_derivatives(*wavefunctions[it.second.front()].pos);
    }
    for (auto &jt : it.second) {
      Wavefunction_Type& wavefunction = wavefunctions[jt];
      auto code = jt & WT::mask;
      switch (code) {
        case WT::normal: basis.host_psi_get(wavefunction, *wavefunction.pos); break;
        case WT::dx: basis.host_psi_get_dx(wavefunction, *wavefunction.pos); break;
        case WT::dy: basis.host_psi_get_dy(wavefunction, *wavefunction.pos); break;
        case WT::dz: basis.host_psi_get_dz(wavefunction, *wavefunction.pos); break;
      }
      wavefunction.ao_to_mo(basis.ao);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void QC_monte<Container, Allocator>::print_mc_head(std::chrono::system_clock::time_point mc_start) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_start);
  std::cout << "Begining MC run at: " << ctime(&tt);
  std::cout.flush();
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void QC_monte<Container, Allocator>::print_mc_tail(double time_span, std::chrono::system_clock::time_point mc_end) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_end);
  std::cout << "Finished MC run at: " << ctime(&tt);
  std::cout << "Spent " << time_span << " second preforming MC integration" << std::endl;
}


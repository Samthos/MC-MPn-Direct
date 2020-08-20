#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <bitset>

#include <unordered_map>

#include "qc_monte.h"

template <typename Container>
QC_monte<Container>::QC_monte(MPI_info p0, IOPs p1, Molecule p2, Basis_Host p3) :
    mpi_info(p0),
    iops(p1),
    molec(p2),
    basis(p3),
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

template <typename Container>
QC_monte<Container>::~QC_monte() {
  delete tau;
  delete electron_pair_list;
}

template <typename Container>
void QC_monte<Container>::move_walkers() {
  electron_pair_list->move(random, electron_pair_weight);
  if (electron_list != nullptr) {
    electron_list->move(random, electron_weight);
  }
}

template <typename Container>
void QC_monte<Container>::print_mc_head(std::chrono::system_clock::time_point mc_start) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_start);
  std::cout << "Begining MC run at: " << ctime(&tt);
  std::cout.flush();
}

template <typename Container>
void QC_monte<Container>::print_mc_tail(double time_span, std::chrono::system_clock::time_point mc_end) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_end);
  std::cout << "Finished MC run at: " << ctime(&tt);
  std::cout << "Spent " << time_span << " second preforming MC integration" << std::endl;
}



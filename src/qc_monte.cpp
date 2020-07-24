#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <bitset>

#include <unordered_map>

#include "qc_monte.h"


template <class Container>
QC_monte<Container>::QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis_Host p3) :
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

  auto movecs = create_movec_parser(iops, mpi_info, molec);

  iocc1 = movecs->iocc1;
  iocc2 = movecs->iocc2;
  ivir1 = movecs->ivir1;
  ivir2 = movecs->ivir2;

  electron_pair_list = create_electron_pair_sampler(iops, molec, electron_pair_weight);
  wavefunctions.emplace(WC::electron_pairs_1, Wavefunction(&electron_pair_list->pos1, movecs));
  wavefunctions.emplace(WC::electron_pairs_2, Wavefunction(&electron_pair_list->pos2, movecs));

  electron_list = nullptr;
  if (iops.iopns[KEYS::TASK] & TASK::ANY_F12) {
    electron_list = create_electron_sampler(iops, molec, electron_weight);
    wavefunctions.emplace(WC::electrons, Wavefunction(&electron_list->pos, movecs));
    if (iops.iopns[KEYS::TASK] & TASK::ANY_F12_VBX) {
      wavefunctions.emplace(WC::electrons_dx, Wavefunction(&electron_list->pos, movecs));
      wavefunctions.emplace(WC::electrons_dy, Wavefunction(&electron_list->pos, movecs));
      wavefunctions.emplace(WC::electrons_dz, Wavefunction(&electron_list->pos, movecs));

      wavefunctions.emplace(WC::electron_pairs_1_dx, Wavefunction(&electron_pair_list->pos1, movecs));
      wavefunctions.emplace(WC::electron_pairs_1_dy, Wavefunction(&electron_pair_list->pos1, movecs));
      wavefunctions.emplace(WC::electron_pairs_1_dz, Wavefunction(&electron_pair_list->pos1, movecs));

      wavefunctions.emplace(WC::electron_pairs_2_dx, Wavefunction(&electron_pair_list->pos2, movecs));
      wavefunctions.emplace(WC::electron_pairs_2_dy, Wavefunction(&electron_pair_list->pos2, movecs));
      wavefunctions.emplace(WC::electron_pairs_2_dz, Wavefunction(&electron_pair_list->pos2, movecs));
    }
  }
  for (auto &it : wavefunctions) {
    wavefunction_groups[it.first & WS::mask].push_back(it.first);
  }

  //initialize walkers
  tau = create_tau_sampler(iops, movecs);
}

template <class Container>
QC_monte<Container>::~QC_monte() {
  delete tau;
  delete electron_pair_list;
}

template <class Container>
void QC_monte<Container>::move_walkers() {
  electron_pair_list->move(random, electron_pair_weight);
  if (electron_list != nullptr) {
    electron_list->move(random, electron_weight);
  }
}

template <class Container>
void QC_monte<Container>::print_mc_head(std::chrono::system_clock::time_point mc_start) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_start);
  std::cout << "Begining MC run at: " << ctime(&tt);
  std::cout.flush();
}

template <class Container>
void QC_monte<Container>::print_mc_tail(double time_span, std::chrono::system_clock::time_point mc_end) {
  std::time_t tt = std::chrono::system_clock::to_time_t(mc_end);
  std::cout << "Finished MC run at: " << ctime(&tt);
  std::cout << "Spent " << time_span << " second preforming MC integration" << std::endl;
}

template <class Container>
Energy<Container>::Energy(MPI_info p1, IOPs p2, Molec p3, Basis_Host p4) : QC_monte(p1, p2, p3, p4) {
  int max_tau_coordinates = 0;
  int total_control_variates = 0;

  if (iops.iopns[KEYS::TASK] & TASK::MP2) {
    energy_functions.push_back(create_MCMP2(iops.iopns[KEYS::MP2CV_LEVEL]));
  }
  if (iops.iopns[KEYS::TASK] & TASK::MP3) {
    energy_functions.push_back(create_MCMP3(iops.iopns[KEYS::MP3CV_LEVEL]));
  }
  if (iops.iopns[KEYS::TASK] & TASK::MP4) {
    energy_functions.push_back(create_MCMP4(iops.iopns[KEYS::MP4CV_LEVEL], electron_pair_list));
  }
  if (iops.iopns[KEYS::TASK] & TASK::MP2_F12_V) {
    energy_functions.push_back(new MP2_F12_V(iops));
  }
  if (iops.iopns[KEYS::TASK] & TASK::MP2_F12_VBX) {
    energy_functions.push_back(new MP2_F12_VBX(iops));
  }

  emp.resize(energy_functions.size());
  for (auto &it : energy_functions) {
    control.emplace_back(it->n_control_variates);
    cv.push_back(create_accumulator(electron_pair_list->requires_blocking(), std::vector<double>(it->n_control_variates, 0.0)));

    max_tau_coordinates = std::max(max_tau_coordinates, it->n_tau_coordinates);
    total_control_variates += it->n_control_variates;
  }

  tau->resize(max_tau_coordinates);
  ovps.init(max_tau_coordinates, iops.iopns[KEYS::ELECTRON_PAIRS]);
  
  control.emplace_back(total_control_variates);
  cv.push_back(create_accumulator(electron_pair_list->requires_blocking(), std::vector<double>(total_control_variates, 0.0)));
}

template <class Container>
Energy<Container>::~Energy() {
  for (auto &item : cv) {
    delete item;
  }
  for (auto &item : energy_functions) {
    delete item;
  }
}

#ifdef HAVE_CUDA
GPU_Energy::GPU_Energy(MPI_info p1, IOPs p2, Molec p3, Basis_Host p4) : Energy(p1, p2, p3, p4) {
  ovps_device.init(ovps.o_set.size(), iops.iopns[KEYS::ELECTRON_PAIRS]);
}
#endif


Dimer::Dimer(MPI_info p1, IOPs p2, Molec p3, Basis_Host p4) : Energy(p1, p2, p3, p4),
                                                         l_emp(emp),
                                                         l_control(control)
{
  Molec monomer_a_geometry;
  Molec monomer_b_geometry;

  monomer_a_geometry.read(mpi_info, iops.sopns[KEYS::MONOMER_A_GEOM]);
  monomer_b_geometry.read(mpi_info, iops.sopns[KEYS::MONOMER_B_GEOM]);

  auto monomer_a_movecs = create_movec_parser(iops, mpi_info, monomer_a_geometry, KEYS::MONOMER_A_MOVECS, MOVEC_TYPE::NWCHEM);
  auto monomer_b_movecs = create_movec_parser(iops, mpi_info, monomer_b_geometry, KEYS::MONOMER_B_MOVECS, MOVEC_TYPE::NWCHEM);

  monomer_a_tau = create_tau_sampler(iops, monomer_a_movecs);
  monomer_b_tau = create_tau_sampler(iops, monomer_b_movecs);

  monomer_a_tau->resize(tau->get_n_coordinates());
  monomer_b_tau->resize(tau->get_n_coordinates());

  for (auto &it : wavefunctions) {
    auto pos_source = it.second.pos;
    monomer_a_wavefunctions.emplace(it.first, Wavefunction(pos_source, monomer_a_movecs));
    monomer_b_wavefunctions.emplace(it.first, Wavefunction(pos_source, monomer_b_movecs));
  }
}

Dimer::~Dimer() {
}

/*
Dimer::Dimer(MPI_info p1, IOPs p2, Molec p3, Basis p4) : QC_monte(p1, p2, p3, p4) {
  // build temp list of current wavefunctions keys

  // loop over wavefunctions keys.
  //     wavefunctions.emplace(monomer_a & key, wavefunction(monomer_a))
  //     wavefunctions.emplace(monomer_b & key, wavefunction(monomer_b))
  //     wavefunction_groups[key & mask].push_back(monomer_a & key)
  //     wavefunction_groups[key & mask].push_back(monomer_b & key)

  // ^^^^need to update WT::mask with new dimer/monomer keys

  // create extra tau functions
}
*/

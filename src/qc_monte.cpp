#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <unordered_map>

#include "qc_monte.h"


QC_monte::QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis p3) :
    mpi_info(p0),
    iops(p1),
    molec(p2),
    basis(p3),
    electron_pair_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    electron_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    random(iops.iopns[KEYS::DEBUG])
{
  numBand = iops.iopns[KEYS::NUM_BAND];
  offBand = iops.iopns[KEYS::OFF_BAND];
  nDeriv = iops.iopns[KEYS::DIFFS];

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;

  electron_pair_list = create_electron_pair_sampler(iops, molec, electron_pair_weight);
  wavefunctions.emplace(WC::electron_pairs_1, Wavefunction(&electron_pair_list->pos1, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
  wavefunctions.emplace(WC::electron_pairs_2, Wavefunction(&electron_pair_list->pos2, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));

  electron_list = nullptr;
  if (iops.iopns[KEYS::TASK] & TASK::ANY_F12) {
    electron_list = create_electron_sampler(iops, molec, electron_weight);
    wavefunctions.emplace(WC::electrons, Wavefunction(&electron_list->pos, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
    if (iops.iopns[KEYS::TASK] & TASK::ANY_F12_VBX) {
      wavefunctions.emplace(WC::electrons_dx, Wavefunction(&electron_list->pos, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electrons_dy, Wavefunction(&electron_list->pos, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electrons_dz, Wavefunction(&electron_list->pos, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));

      wavefunctions.emplace(WC::electron_pairs_1_dx, Wavefunction(&electron_pair_list->pos1, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electron_pairs_1_dy, Wavefunction(&electron_pair_list->pos1, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electron_pairs_1_dz, Wavefunction(&electron_pair_list->pos1, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));

      wavefunctions.emplace(WC::electron_pairs_2_dx, Wavefunction(&electron_pair_list->pos2, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electron_pairs_2_dy, Wavefunction(&electron_pair_list->pos2, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
      wavefunctions.emplace(WC::electron_pairs_2_dz, Wavefunction(&electron_pair_list->pos2, basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2));
    }
  }

  //initialize walkers
  basis.gpu_alloc(iops.iopns[KEYS::ELECTRON_PAIRS], molec);
  tau = create_tau_sampler(iops, basis);
}

QC_monte::~QC_monte() {
  basis.gpu_free();
  delete tau;
  delete electron_pair_list;
}

void QC_monte::move_walkers() {
  electron_pair_list->move(random, electron_pair_weight);
  if (electron_list != nullptr) {
    electron_list->move(random, electron_weight);
  }
}

void QC_monte::print_mc_head(std::chrono::high_resolution_clock::time_point mc_start) {
  std::time_t tt = std::chrono::high_resolution_clock::to_time_t(mc_start);
  std::cout << "Begining MC run at: " << ctime(&tt);
  std::cout.flush();
}

void QC_monte::print_mc_tail(double time_span, std::chrono::high_resolution_clock::time_point mc_end) {
  std::time_t tt = std::chrono::high_resolution_clock::to_time_t(mc_end);
  std::cout << "Finished MC run at: " << ctime(&tt);
  std::cout << "Spent " << time_span << " second preforming MC integration" << std::endl;
}

Energy::Energy(MPI_info p1, IOPs p2, Molec p3, Basis p4) : QC_monte(p1, p2, p3, p4) {
  int max_tau_coordinates = 0;
  int total_control_variates = 0;

  std::cout << iops.iopns[KEYS::TASK] << "\n";
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
    energy_functions.push_back(new MP2_F12_V(iops, basis));
  }

  emp.resize(energy_functions.size());
  for (auto &it : energy_functions) {
    control.emplace_back(it->n_control_variates);
    cv.push_back(create_accumulator(electron_pair_list->requires_blocking(), std::vector<double>(it->n_control_variates, 0.0)));

    max_tau_coordinates = std::max(max_tau_coordinates, it->n_tau_coordinates);
    total_control_variates += it->n_control_variates;
  }

  tau->resize(max_tau_coordinates);
  ovps.init(max_tau_coordinates, iops.iopns[KEYS::ELECTRON_PAIRS], basis);
  
  control.emplace_back(total_control_variates);
  cv.push_back(create_accumulator(electron_pair_list->requires_blocking(), std::vector<double>(total_control_variates, 0.0)));
}

Energy::~Energy() {
  for (auto &item : cv) {
    delete item;
  }
  for (auto &item : energy_functions) {
    delete item;
  }
}

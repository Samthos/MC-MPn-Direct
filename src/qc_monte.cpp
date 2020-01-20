#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "qc_monte.h"


QC_monte::QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis p3) :
    mpi_info(p0),
    iops(p1),
    molec(p2),
    basis(p3),
    electron_pair_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    electron_weight(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]),
    electron_pair_psi1(iops.iopns[KEYS::MC_NPAIR], basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2),
    electron_pair_psi2(iops.iopns[KEYS::MC_NPAIR], basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2),
    electron_psi(iops.iopns[KEYS::ELECTRONS], basis.iocc1, basis.iocc2, basis.ivir1, basis.ivir2),
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
  electron_list = nullptr;

  //initialize walkers
  basis.gpu_alloc(iops.iopns[KEYS::MC_NPAIR], molec);
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

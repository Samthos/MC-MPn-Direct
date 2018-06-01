#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "qc_monte.h"

QC_monte::QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis p3, GTO_Weight p4) : mpi_info(p0), iops(p1), molec(p2), basis(p3), mc_basis(p4) {
  random.seed(iops.iopns[KEYS::DEBUG]);

  numBand = iops.iopns[KEYS::NUM_BAND];
  offBand = iops.iopns[KEYS::OFF_BAND];
  nBlock = iops.iopns[KEYS::NBLOCK];
  nDeriv = iops.iopns[KEYS::DIFFS];

  iocc1 = basis.iocc1;
  iocc2 = basis.iocc2;
  ivir1 = basis.ivir1;
  ivir2 = basis.ivir2;

  //initialize walkers
  el_pair_list.resize(iops.iopns[KEYS::MC_NPAIR]);
  for (auto &it : el_pair_list) {
    it.init(ivir2);
  }

  //init wavefunctions walkers
  for (auto &it : el_pair_list) {
    it.is_new = true;
  }

  basis.gpu_alloc(iops.iopns[KEYS::MC_NPAIR], molec);
}

void QC_monte::move_walkers() {
  for (auto &it : el_pair_list) {
    it.mc_move_scheme(random, molec, mc_basis);
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

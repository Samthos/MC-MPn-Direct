//
// Created by aedoran on 6/1/18.
//
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifdef HAVE_MPII
#include "mpi.h"
#endif

#include "../qc_monte.h"

void GF::monte_energy() {
  int checkNum = 1;
  int print_mat;

  //clock
  std::chrono::high_resolution_clock::time_point step_start, step_end, mc_start, mc_end;
  std::chrono::duration<double> time_span{};

  //start clock and print out time
  if (mpi_info.sys_master) {
    step_start = std::chrono::high_resolution_clock::now();
    mc_start = std::chrono::high_resolution_clock::now();
    print_mc_head(mc_start);
  }

  for (auto step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // move walkers
    move_walkers();

    // calcalate new wave function
    update_wavefunction();

    // generate new tau values
    tau->new_tau(random);

    //calculate energies
    mc_local_energy(step);
    for (auto & qep : qeps) {
      qep.blockIt(step);
    }

    // print every 128 steps
    if (0 == step % 128) {
      // Reduce variables across all threads
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      for (auto & qep : qeps) {
        qep.reduce();
      }

      if (mpi_info.sys_master) {  //print out results
        step_end = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start);
        step_start = std::chrono::high_resolution_clock::now();
        for (auto & qep : qeps) {
          qep.print(step, time_span.count());
        }
      }
    }

    if ((iops.iopns[KEYS::TASK] == TASKS::GFFULL) || (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF)) {
      mc_end = std::chrono::high_resolution_clock::now();
      time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
      print_mat = (time_span.count() > checkNum * 900);
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&print_mat, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      if (print_mat) {
        checkNum = full_print(step, checkNum);
#ifdef HAVE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
    }
  }

  if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    full_print(iops.iopns[KEYS::MC_TRIAL], 2);
  }

  //close streams and print out time
  if (mpi_info.sys_master) {
    mc_end = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
    print_mc_tail(time_span.count(), mc_end);
  }
}

void GF2::mc_local_energy(const int& step) {
  if (iops.iopns[KEYS::TASK] == TASKS::GF || iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
    for (auto &it : qeps[0].qeps) {
      std::fill(it.begin(), it.end(), 0.00);
    }
  }

  ovps.update_ovps(basis.h_basis, el_pair_list, tau);
  ovps.update_ovps_02(basis.h_basis);
  mcgf2_local_energy_core();
  int band = 0;
  if (iops.iopns[KEYS::TASK] == TASKS::GF) {
    mcgf2_local_energy(qeps[0].qeps);
  } else if (iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
    mcgf2_local_energy_diff(qeps[0].qeps);
  } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL) {
      mcgf2_local_energy_full(band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      mcgf2_local_energy_full_diff(band);
      if (step > 0) {
      // mc_gf2_statistics(band, step);
      std::cerr << "no full diffs\n";
      }
    }
  if (step > 0 && (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF || iops.iopns[KEYS::TASK] == TASKS::GFFULL)) {
    mc_gf_statistics(step, ovps.d_ovps.enBlock, ovps.d_ovps.enEx1, ovps.d_ovps.enCov);
  }
}

int GF2::full_print(int& step, int checkNum) {
  for (auto band = 0; band < numBand; band++) {
    mc_gf_full_print(band, step, checkNum % 2, 0, ovps.d_ovps.enEx1, ovps.d_ovps.enCov);
  }
  checkNum += 1;
  return checkNum;
}

void GF3::mc_local_energy(const int& step) {
  if (iops.iopns[KEYS::TASK] == TASKS::GF || iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
    for (auto &it : qeps[0].qeps) {
      std::fill(it.begin(), it.end(), 0.00);
    }
    for (auto &it : qeps[1].qeps) {
      std::fill(it.begin(), it.end(), 0.00);
    }
  } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    ovps.zero_energy_arrays_03();
  }

  ovps.update_ovps_03(el_pair_list->data(), tau);

  mcgf2_local_energy_core();
  mcgf3_local_energy_core();
  for (int band = 0; band < numBand; band++) {
    if (iops.iopns[KEYS::TASK] == TASKS::GF) {
      mcgf2_local_energy(qeps[0].qeps);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
      mcgf2_local_energy_diff(qeps[0].qeps);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL) {
      mcgf2_local_energy_full(band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      mcgf2_local_energy_full_diff(band);
      std::cerr << "no full diffs\n";
    }

    if (iops.iopns[KEYS::TASK] == TASKS::GF) {
      mcgf3_local_energy(qeps[1].qeps[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
      mcgf3_local_energy_diff(qeps[1].qeps[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL) {
      mcgf3_local_energy_full(band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      mcgf3_local_energy_full_diff(band);
      std::cerr << "no full diffs\n";
    }
  }

  if (step > 0 && (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF || iops.iopns[KEYS::TASK] == TASKS::GFFULL)) {
    mc_gf_statistics(step, ovps.d_ovps.enBlock, ovps.d_ovps.enEx1, ovps.d_ovps.enCov);
  }
}

int GF3::full_print(int& step, int checkNum) {
  for (auto band = 0; band < numBand; band++) {
    mc_gf_full_print(band, step, checkNum % 2, 0, ovps.d_ovps.enEx1, ovps.d_ovps.enCov);
  }
  checkNum += 1;
  return checkNum;
}

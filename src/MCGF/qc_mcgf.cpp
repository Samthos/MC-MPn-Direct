//
// Created by aedoran on 6/1/18.
//
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../qc_mpi.h"
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

  tau->new_tau(random);
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
      MPI_info::barrier();
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

    if ((iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL) || (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF)) {
      mc_end = std::chrono::high_resolution_clock::now();
      time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
      print_mat = (time_span.count() > checkNum * 900);
      MPI_info::barrier();
      MPI_info::broadcast_int(&print_mat, 1);
      if (print_mat) {
        checkNum = full_print(step, checkNum);
        MPI_info::barrier();
      }
    }
  }

  if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
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
  for (auto &it : qeps[0].qeps) {
    std::fill(it.begin(), it.end(), 0.00);
  }

  ovps.update_ovps(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], tau);
  mcgf2_local_energy_core();

  for(int band=0;band<numBand;band++) {
    if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL) {
      mcgf2_local_energy_full(band);
    } else if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
      mcgf2_local_energy_full_diff(band);
      if (step > 0) {
        mc_gf_full_diffs(band, {tau->get_tau(0), -tau->get_tau(0)});
      }
    }
  }
  if (step > 0 && (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL)) {
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
  for (auto &it : qeps[0].qeps) {
    std::fill(it.begin(), it.end(), 0.00);
  }
  for (auto &it : qeps[1].qeps) {
    std::fill(it.begin(), it.end(), 0.00);
  }

  ovps.update_ovps(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], tau);

  mcgf2_local_energy_core();
  mcgf3_local_energy_core();

  if(iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GF || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL) {
    mcgf3_local_energy(qeps[1].qeps);
  } else if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFDIFF || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
    mcgf3_local_energy_diff(qeps[1].qeps);
  }

  for (int band = 0; band < numBand; band++) {
    if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL) {
      mcgf3_local_energy_full(band);
    } else if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
      mcgf3_local_energy_full_diff(band);
        if (step > 0) {
          mc_gf_full_diffs(band, {tau->get_tau(0), -tau->get_tau(0), tau->get_tau(1), -tau->get_tau(1), 
                                  tau->get_tau(0) + tau->get_tau(1), -tau->get_tau(0) - tau->get_tau(1), 
                                  0.0});
        }
    }
  }

  if (step > 0 && (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL)) {
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

Diagonal_GF::Diagonal_GF(MPI_info p1, IOPs p2, Molec p3, Basis p4)
  : GF(p1, p2, p3, p4),
  gf2_functional(p2, p4)
{
  int max_tau_coordinates = 1;

  qeps.emplace_back(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, iops.sopns[KEYS::JOBNAME], 2);

  tau->resize(2);
  // tau->resize(max_tau_coordinates);
  ovps.init(max_tau_coordinates, iops.iopns[KEYS::ELECTRON_PAIRS], basis);
}

void Diagonal_GF::monte_energy() {
  // clock
  std::chrono::high_resolution_clock::time_point step_start, step_end, mc_start, mc_end;
  std::chrono::duration<double> time_span{};

  // start clock and print out time
  if (mpi_info.sys_master) {
    step_start = std::chrono::high_resolution_clock::now();
    mc_start = std::chrono::high_resolution_clock::now();
    print_mc_head(mc_start);
  }

  tau->new_tau(random);
  for (auto step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // move walkers
    move_walkers();

    // calcalate new wave function
    update_wavefunction();

    // generate new tau values
    tau->new_tau(random);

    for (auto &qep : qeps) {
      for (auto &it : qep.qeps) {
        std::fill(it.begin(), it.end(), 0.00);
      }
    }
    //calculate energies
    mc_local_energy(step);
    for (auto & qep : qeps) {
      qep.blockIt(step);
    }

    // print every 128 steps
    if (0 == step % 128) {
      // Reduce variables across all threads
      MPI_info::barrier();
      for (auto & qep : qeps) {
        qep.reduce();
      }

      if (mpi_info.sys_master) {  // print out results
        step_end = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start);
        step_start = std::chrono::high_resolution_clock::now();
        for (auto & qep : qeps) {
          qep.print(step, time_span.count());
        }
      }
    }
  }

  // close streams and print out time
  if (mpi_info.sys_master) {
    mc_end = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
    print_mc_tail(time_span.count(), mc_end);
  }
}

void Diagonal_GF::mc_local_energy(const int& step) {
  ovps.update_ovps(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], tau);
  gf2_functional.energy(qeps[0].qeps, wavefunctions, ovps, electron_pair_list, tau);
}

int Diagonal_GF::full_print(int& step, int checkNum) {
}

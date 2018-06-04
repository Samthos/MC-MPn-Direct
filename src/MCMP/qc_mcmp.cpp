//
// Created by aedoran on 6/1/18.
//

//
// Created by aedoran on 6/1/18.
//
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>


#include "../qc_monte.h"
#include "../control_variate.h"
#include "../timer.h"

void MP2::monte_energy() {
  // variables to store emp2 energy
  double emp2;
  std::vector<double> control(2);
  ControlVariate cv_emp2(2, {0, 0});

  std::ofstream out_mp2;
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();

    std::string filename = iops.sopns[KEYS::JOBNAME] + ".22";
    out_mp2.open(filename.c_str());
  }

  // --- initialize
  update_wavefunction();
  mcmp2_energy(emp2, control);
  for (int step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    for (auto& it : el_pair_list) {
      it.mc_move_scheme(random, molec, mc_basis);
    }

    // update wavefunction
    update_wavefunction();

    // calcaulte energy for step
    mcmp2_energy(emp2, control);

    // accumulate
    cv_emp2.add(emp2, control);

    // print if i is a multiple of 128
    if (0 == step % 128) {
      out_mp2 << cv_emp2 << "\t";
      out_mp2 << stepTimer << "\n";
      stepTimer.Start();
    }
  }
  if (mpi_info.sys_master) {
    std::cout << "Spent " << mcTimer << " second preforming MC integration" << std::endl;
    out_mp2.close();
  }
}

void MP2::new_tau() {
  double p = random.get_rand();
  double tau = -log(1.0 - p) / lambda;
  tau_wgt = 1.0 / (lambda * (1.0 - p));

  for (int jt = iocc1; jt < iocc2; ++jt) {
    tau_values[jt] = exp(basis.nw_en[jt] * tau);
  }
  for (int jt = ivir1; jt < ivir2; ++jt) {
    tau_values[jt] = exp(-basis.nw_en[jt] * tau);
  }
}

void MP3::monte_energy() {
  // variables to store emp2 energy
  double emp2, emp3;
  std::vector<double> mp2_control(2), mp3_control(12);
  ControlVariate cv_emp2(2, {0, 0}), cv_emp3(12, {0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0});

  std::ofstream out_mp2, out_mp3;
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();

    std::string filename = iops.sopns[KEYS::JOBNAME] + ".22";
    out_mp2.open(filename.c_str());

    filename = iops.sopns[KEYS::JOBNAME] + ".23";
    out_mp3.open(filename.c_str());
  }

  // --- initialize
  update_wavefunction();
  ovps.new_tau_03(basis, random);
  ovps.update_ovps_03(el_pair_list.data());
  mcmp2_energy(emp2, mp2_control);
  mcmp3_energy(emp3, mp3_control);

  // run monte carlo simulation
  for (int step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    for (auto& it : el_pair_list) {
      it.mc_move_scheme(random, molec, mc_basis);
    }

    // generate new tau values
    ovps.new_tau_03(basis, random);

    // update wavefunction and green's function traces
    update_wavefunction();
    ovps.update_ovps_03(el_pair_list.data());

    // calcaulte energy for step
    mcmp2_energy(emp2, mp2_control);
    mcmp3_energy(emp3, mp3_control);

    // accumulate
    cv_emp2.add(emp2, mp2_control);
    cv_emp3.add(emp3, mp3_control);

    // print if i is a multiple of 128
    if (0 == step % 128) {
      out_mp2 << cv_emp2 << "\t";
      out_mp2 << stepTimer << "\n";
      out_mp2.flush();

      out_mp3 << cv_emp3 << "\t";
      out_mp3 << stepTimer << "\n";
      out_mp3.flush();
      stepTimer.Start();
    }
  }
  if (mpi_info.sys_master) {
    std::cout << "Spent " << mcTimer << " second preforming MC integration" << std::endl;
    out_mp2.close();
    out_mp3.close();
  }
}
/*
void MP3::monte_energy() {
  int checkNum = 1;
  int i, band, print_mat;

  GFStats qeps2(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, nBlock, iops.sopns[KEYS::JOBNAME], 2);
  GFStats qeps3(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, nBlock, iops.sopns[KEYS::JOBNAME], 3);

  //clock
  std::chrono::high_resolution_clock::time_point step_start, step_end, mc_start, mc_end;
  std::chrono::duration<double> time_span;

  //start clock and print out time
  if (mpi_info.sys_master) {
    step_start = std::chrono::high_resolution_clock::now();
    mc_start = std::chrono::high_resolution_clock::now();
    print_mc_head(mc_start);
  }

  mc_local_energy(qeps2.qeps, qeps3.qeps, 0);
  for (i = 1; i <= iops.iopns[KEYS::MC_TRIAL]; i++) {
    move_walkers();

    //calculate energies
    mc_local_energy(qeps2.qeps, qeps3.qeps, i);
    qeps2.blockIt(i);
    qeps3.blockIt(i);

//print every 128 steps
#ifdef QUAD_TAU
    if (0 == i % 16)
#else
    if (0 == i % 128)
#endif
    {
      //Reduce variables across all threads
      MPI_Barrier(MPI_COMM_WORLD);
      qeps2.reduce();
      qeps3.reduce();

      if (mpi_info.sys_master) {  //print out results
        step_end = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start);
        step_start = std::chrono::high_resolution_clock::now();
        qeps2.print(i, time_span.count());
        qeps3.print(i, time_span.count());
      }
    }

    if ((iops.iopns[KEYS::TASK] == TASKS::GFFULL) || (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF)) {
      mc_end = std::chrono::high_resolution_clock::now();
      time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
      print_mat = (time_span.count() > checkNum * 900);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&print_mat, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (print_mat) {
        for (band = 0; band < numBand; band++) {
          mc_gf2_full_print(band, i, checkNum % 2);
          mc_gf3_full_print(band, i, checkNum % 2);
        }
        checkNum++;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }

  if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    for (i = 0; i < numBand; i++) {
      mc_gf2_full_print(i, iops.iopns[KEYS::MC_TRIAL], 2);
      std::cout.flush();
      mc_gf3_full_print(i, iops.iopns[KEYS::MC_TRIAL], 2);
      std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //close streams and print out time
  if (mpi_info.sys_master) {
    mc_end = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
    print_mc_tail(time_span.count(), mc_end);
  }
}

void MP3::mc_local_energy(std::vector<std::vector<double>> &qeps2, std::vector<std::vector<double>> &qeps3, int step) {
  update_wavefunction();

  if (iops.iopns[KEYS::TASK] == TASKS::GF || iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
    for (auto &it : qeps2) {
      std::fill(it.begin(), it.end(), 0.00);
    }
    for (auto &it : qeps3) {
      std::fill(it.begin(), it.end(), 0.00);
    }
  } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    ovps.zero_energy_arrays_03();
  }

#ifdef QUAD_TAU
  for (auto it = 0; it < 21; it++) {
    for (auto jt = 0; jt < 21; jt++) {
      ovps.set_tau_03(it, jt);
#else
  ovps.new_tau_03(basis, random);
#endif

  ovps.update_ovps_03(el_pair_list.data());
#ifdef QUAD_TAU
  if (0 == jt) {
#endif
  mcgf2_local_energy_core();
#ifdef QUAD_TAU
  }
#endif
  mcgf3_local_energy_core();

  for (int band = 0; band < numBand; band++) {
#ifdef QUAD_TAU
    if (0 == jt) {
#endif
    if (iops.iopns[KEYS::TASK] == TASKS::GF) {
      mcgf2_local_energy(qeps2[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
      mcgf2_local_energy_diff(qeps2[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL) {
      mcgf2_local_energy_full(band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      mcgf2_local_energy_full_diff(band);
      if (step > 0) {
        mc_gf2_statistics(band, step);
      }
    }
#ifdef QUAD_TAU
    }
#endif

    if (iops.iopns[KEYS::TASK] == TASKS::GF) {
      mcgf3_local_energy(qeps3[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
      mcgf3_local_energy_diff(qeps3[band], band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL) {
      mcgf3_local_energy_full(band);
    } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      mcgf3_local_energy_full_diff(band);
      if (step > 0) {
        mc_gf3_statistics(band, step);
      }
    }
  }
#ifdef QUAD_TAU
  }
  }
#endif

  if (step > 0 && (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF || iops.iopns[KEYS::TASK] == TASKS::GFFULL)) {
    mc_gf_statistics(step, qeps2, ovps.d_ovps.en2, ovps.d_ovps.en2Block, ovps.d_ovps.en2Ex1, ovps.d_ovps.en2Ex2);
    mc_gf_statistics(step, qeps3, ovps.d_ovps.en3, ovps.d_ovps.en3Block, ovps.d_ovps.en3Ex1, ovps.d_ovps.en3Ex2);
  }
}
*/

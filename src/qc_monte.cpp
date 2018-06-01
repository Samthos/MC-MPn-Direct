#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "mpi.h"

#include "qc_monte.h"

QC_monte::QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis p3, MC_Basis p4) :
    mpi_info(p0), iops(p1), molec(p2), basis(p3), mc_basis(p4) {
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
  for(auto &it : el_pair_list) {
    it.init(ivir2);
    it.pos_init(molec, random);
    it.weight_func_set(molec, mc_basis);
  }
  //Perform Metropolis Monte Carlo burn in
  nsucc = 0;
  nfail = 0;
  delx    = iops.dopns[KEYS::MC_DELX];
  for(int i = 1; i <= 100000; i++) {
    move_walkers();
    if(0 == i%1000) {
      scale_delx();
    }
  }
  nsucc = 0;
  nfail = 0;
  //init wavefunctions walkers
  for(auto &it : el_pair_list) {
    it.is_new = true; 
  }

  basis.gpu_alloc(iops.iopns[KEYS::MC_NPAIR], molec);
}
void QC_monte::move_walkers() {
  for (auto &it : el_pair_list) {
    it.mc_move_scheme(&nsucc, &nfail, delx, random, molec, mc_basis);
  }
}
void QC_monte::scale_delx() {
  double ratio = ((double) nfail)/((double) (nfail + nsucc));
  if (ratio < 0.5) {
    ratio = std::min(1.0/(2.0*ratio), 1.1);
  } else {
    ratio = std::max(0.9, 1.0/(2.0*ratio));
  }
  delx = delx * ratio; //1.1

  nsucc = 0;
  nfail = 0;
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

void QC_Monte_2::monte_energy() {
  int checkNum = 1;
  int i, band, print_mat;

  GFStats qeps2(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, nBlock, iops.sopns[KEYS::JOBNAME], 2);

  //clock
  std::chrono::high_resolution_clock::time_point step_start, step_end, mc_start, mc_end;
  std::chrono::duration<double> time_span;

  //start clock and print out time 
  if (mpi_info.sys_master) {
    step_start = std::chrono::high_resolution_clock::now();
    mc_start = std::chrono::high_resolution_clock::now();
    print_mc_head(mc_start);
  }

  mc_local_energy(qeps2.qeps, 0);
  for(i = 1; i<= iops.iopns[KEYS::MC_TRIAL]; i++) {
    move_walkers();
    if(0 == i%1000) {
      scale_delx();
    }
  
    //calculate energies
    mc_local_energy(qeps2.qeps, i);
    qeps2.blockIt(i);

    //print every 128 steps
#ifdef QUAD_TAU
    if(0 == i%16)
#else
    if(0 == i%128)
#endif
    {
      //Reduce variables across all threads
      MPI_Barrier(MPI_COMM_WORLD);
      qeps2.reduce();

      if (mpi_info.sys_master) { //print out results
        step_end = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(step_end - step_start);
        step_start = std::chrono::high_resolution_clock::now();
        qeps2.print(i, time_span.count());
      }
    }

    if ((iops.iopns[KEYS::TASK] == TASKS::GFFULL) || (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF)) {
      mc_end = std::chrono::high_resolution_clock::now();
      time_span = std::chrono::duration_cast<std::chrono::duration<double>>(mc_end - mc_start);
      print_mat = (time_span.count() > checkNum * 900);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&print_mat, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (print_mat) {
        for(band=0;band<numBand;band++) {
          mc_gf2_full_print(band, i, checkNum % 2);
        }
        checkNum++;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }  

  if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    for(i=0;i<numBand;i++) {
      mc_gf2_full_print(i, iops.iopns[KEYS::MC_TRIAL], 2);
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

void QC_Monte_2::mc_local_energy(std::vector<std::vector<double>>& qeps2, int step) {
  update_wavefunction();

  if(iops.iopns[KEYS::TASK] == TASKS::GF || iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
    for (auto &it : qeps2) {
      std::fill(it.begin(), it.end(), 0.00);
    }
  } else if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    ovps.zero_energy_arrays_02();
  }

#ifdef QUAD_TAU
  for (auto it = 0; it < 21; it++) {
      ovps.set_tau_02(it);
#else
  ovps.new_tau_02(basis, random);
#endif

  ovps.update_ovps_02(el_pair_list.data());
  mcgf2_local_energy_core();
  for(int band=0;band<numBand;band++) {
    if(iops.iopns[KEYS::TASK] == TASKS::GF) {
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
  }
#ifdef QUAD_TAU
  }
#endif

  if (step > 0 && (iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF || iops.iopns[KEYS::TASK] == TASKS::GFFULL)) {
    mc_gf_statistics(step, qeps2, ovps.d_ovps.en2, ovps.d_ovps.en2Block, ovps.d_ovps.en2Ex1, ovps.d_ovps.en2Ex2);
  }
}

void QC_Monte_3::monte_energy() {
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
  for(i = 1; i<= iops.iopns[KEYS::MC_TRIAL]; i++) {
    move_walkers();
    if(0 == i%1000) {
      scale_delx();
    }
  
    //calculate energies
    mc_local_energy(qeps2.qeps, qeps3.qeps, i);
    qeps2.blockIt(i);
    qeps3.blockIt(i);

    //print every 128 steps
#ifdef QUAD_TAU
    if(0 == i%16)
#else
    if(0 == i%128)
#endif
    {
      //Reduce variables across all threads
      MPI_Barrier(MPI_COMM_WORLD);
      qeps2.reduce();
      qeps3.reduce();

      if (mpi_info.sys_master) { //print out results
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
        for(band=0;band<numBand;band++) {
          mc_gf2_full_print(band, i, checkNum % 2);
          mc_gf3_full_print(band, i, checkNum % 2);
        }
        checkNum++;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }  

  if (iops.iopns[KEYS::TASK] == TASKS::GFFULL || iops.iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
    for(i=0;i<numBand;i++) {
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

void QC_Monte_3::mc_local_energy(std::vector<std::vector<double>>& qeps2, std::vector<std::vector<double>>& qeps3, int step) {
  update_wavefunction();

  if(iops.iopns[KEYS::TASK] == TASKS::GF || iops.iopns[KEYS::TASK] == TASKS::GFDIFF) {
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

  for(int band=0;band<numBand;band++) {
    #ifdef QUAD_TAU
    if (0 == jt) {
    #endif
      if(iops.iopns[KEYS::TASK] == TASKS::GF) {
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

    if(iops.iopns[KEYS::TASK] == TASKS::GF) {
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

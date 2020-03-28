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

void MP::monte_energy() {

  std::vector<std::ofstream> output(emp.size() + 1);

#ifdef DIMER_PRINT

  std::vector<std::ofstream> dimer_output(emp.size() + 1);

#endif // DIMER_PRINT

  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();
    print_mc_head(mcTimer.StartTime());

    std::string filename;
    for (auto i = 0; i < emp.size(); i++) {
      filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
      output[i].open(filename.c_str());
    }
    filename = iops.sopns[KEYS::JOBNAME] + ".20";
    output.back().open(filename.c_str());

  // if DIMER_PRINT is defined: open binary ofstream for each process
#ifdef DIMER_PRINT

    std::string dimer_filename;
    for (auto i = 0; i < emp.size(); i++) {
      dimer_filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2) + ".bin";
      dimer_output[i].open(dimer_filename.c_str(), std::ios::binary);
    }
    dimer_filename = iops.sopns[KEYS::JOBNAME] + ".20.bin";
    dimer_output.back().open(dimer_filename.c_str(), std::ios::binary);

#endif // DIMER_PRINT
  }


  // --- initialize
  for (int step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    move_walkers();

    // update wavefunction
    update_wavefunction();

    // zero energy arrarys
    zero_energies();

    // calcaulte energy for step
    do {
      tau->new_tau(random);
      energy();
    } while (tau->next());

#ifdef DIMER_PRINT
    /*
     * dump all values to dimer ofstream
     * values are stored in emp and control
     */
#endif // DIMER_PRINT

    // accumulate
    auto cv_back = control.back().begin();
    for (auto it = 0; it < cv.size()-1; it++) {
      cv[it]->add(emp[it], control[it]);
      std::copy(control[it].begin(), control[it].end(), cv_back);
      cv_back += control[it].size();
      // std::cout << std::distance(control.back().begin(), cv_back) << std::endl;
    }
    cv.back()->add(std::accumulate(emp.begin(), emp.end(), 0.0), control.back());
    // print if i is a multiple of 128
    if (0 == step % 128) {
      for (auto i = 0; i < emp.size(); i++) {
        output[i] << *cv[i] << "\t";
        output[i] << stepTimer << "\n";
        output[i].flush();
      }
      output.back() << *cv.back() << "\t" << stepTimer << "\n";
      output.back().flush();
      stepTimer.Start();
    }
  }

// suppress JSON output if DIMER_PRINT is defined
#ifndef DIMER_PRINT

  for (auto i = 0; i < emp.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
      cv[i]->to_json(filename);
  }
  {
    std::string filename = iops.sopns[KEYS::JOBNAME] + ".20";
    cv.back()->to_json(filename);
  }

#endif // DIMER_PRINT

  if (mpi_info.sys_master) {
    mcTimer.Stop();
    print_mc_tail(mcTimer.Span(), mcTimer.EndTime());
    for (auto i = 0; i < emp.size(); i++) {
      output[i].close();
    }
  }
}

void MP::zero_energies() {
  std::fill(emp.begin(), emp.end(), 0.0);
  for (auto &c : control) {
    std::fill(c.begin(), c.end(), 0.0);
  }
}

void MP2::energy() {
  mcmp2_energy_fast(emp[0], control[0]);
}

void MP3::energy() {
  ovps.update_ovps(electron_pair_psi1, electron_pair_psi2, tau);
  if (tau->is_new(1)) {
    mcmp2_energy(emp[0], control[0]);
  }
  mcmp3_energy(emp[1], control[1]);
}

void MP4::energy() {
  ovps.update_ovps(electron_pair_psi1, electron_pair_psi2, tau);
  if (tau->is_new(1)) {
    mcmp2_energy(emp[0], control[0]);
  }
  if (tau->is_new(2)) {
    mcmp3_energy(emp[1], control[1]);
  }
  mcmp4_energy(emp[2], control[2]);
}

void MP2F12_V::energy() {
  mcmp2_energy_fast(emp[0], control[0]);
  emp[1] = mp2f12_v_engine.calculate_v(electron_pair_psi1, electron_pair_psi2, electron_psi, electron_pair_list, electron_list);
}


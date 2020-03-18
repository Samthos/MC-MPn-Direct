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

void Energy::zero_energies() {
  std::fill(emp.begin(), emp.end(), 0.0);
  for (auto &c : control) {
    std::fill(c.begin(), c.end(), 0.0);
  }
}

void Energy::monte_energy() {
  std::vector<std::ofstream> output(emp.size()+1);
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();
    print_mc_head(mcTimer.StartTime());

    for (auto i = 0; i < energy_functions.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + "." + energy_functions[i]->extension;
      output[i].open(filename.c_str());
    }
    {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".20";
      output.back().open(filename.c_str());
    }
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

  for (auto i = 0; i < emp.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + "." + energy_functions[i]->extension;
      cv[i]->to_json(filename);
  }
  {
    std::string filename = iops.sopns[KEYS::JOBNAME] + ".20";
    cv.back()->to_json(filename);
  }
  if (mpi_info.sys_master) {
    mcTimer.Stop();
    print_mc_tail(mcTimer.Span(), mcTimer.EndTime());
    for (auto i = 0; i < emp.size(); i++) {
      output[i].close();
    }
  }
}

void Energy::energy() {
  ovps.update_ovps(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], tau);
  for (int i = 0; i < energy_functions.size(); i++) {
    if (!energy_functions[i]->is_f12) {
      if (tau->is_new(energy_functions[i]->n_tau_coordinates)) {
        energy_functions[i]->energy(emp[i], control[i], ovps, electron_pair_list, tau);
      }
    } else {
      if (tau->is_new(energy_functions[i]->n_tau_coordinates)) {
        energy_functions[i]->energy_f12(emp[i], control[i], wavefunctions, electron_pair_list, electron_list);
      }
    }
  }
}

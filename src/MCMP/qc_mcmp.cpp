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

void MP::monte_energy() {
  std::vector<std::ofstream> output(emp.size()+1);
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();
    print_mc_head(mcTimer.StartTime());

    for (auto i = 0; i < emp.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
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
#ifndef FULL_PRINTING
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
#else
    output.back() << step << ",";
    for (double i : emp) {
      output.back() << std::setprecision(std::numeric_limits<double>::digits10 + 1)<< i << ",";
    }
    output.back() << "\n";
#endif
  }

  for (auto i = 0; i < emp.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
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
  ovps.update_ovps(basis.h_basis, el_pair_list, tau);
  if (tau->is_new(1)) {
    mcmp2_energy(emp[0], control[0]);
  }
  mcmp3_energy(emp[1], control[1]);
}

void MP4::energy() {
  ovps.update_ovps(basis.h_basis, el_pair_list, tau);
  mcmp2_energy(emp[0], control[0]);
  mcmp3_energy(emp[1], control[1]);
  mcmp4_energy(emp[2], control[2]);
}

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
  // variables to store emp2 energy
//  double emp2;
//  std::vector<double> control(2);
//  ControlVariate cv_emp2(2, {0, 0});

  std::vector<std::ofstream> output(emp.size());
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();

    for (auto i = 0; i < emp.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
      output[i].open(filename.c_str());
    }
  }

  // --- initialize
  energy();
  for (int step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    move_walkers();

    // update wavefunction
    update_wavefunction();

    // calcaulte energy for step
    energy();

    // accumulate
    for (auto it = 0; it < cv.size(); it++) {
      cv[it].add(emp[it], control[it]);
    }

    // print if i is a multiple of 128
    if (0 == step % 128) {
      for (auto i = 0; i < emp.size(); i++) {
        output[i] << cv[i] << "\t";
        output[i] << stepTimer << "\n";
      }
      stepTimer.Start();
    }
  }
  if (mpi_info.sys_master) {
    std::cout << "Spent " << mcTimer << " second preforming MC integration" << std::endl;
    for (auto i = 0; i < emp.size(); i++) {
      output[i].close();
    }
  }
}

void MP2::energy() {
  mcmp2_energy(emp[0], control[0]);
}

void MP3::energy() {
  tau.new_tau(random);
  ovps.update_ovps(&el_pair_list[0], tau);
  mcmp2_energy(emp[0], control[0]);
  mcmp3_energy(emp[1], control[1]);
}

/*
void MP4::monte_energy() {
  // variables to store emp2 energy
  std::vector<double> emp(3);

  std::vector<std::vector<double>> control;
  control.push_back(std::vector<double>(2));
  control.push_back(std::vector<double>(6));
  control.push_back(std::vector<double>(1));

  std::vector<ControlVariate> cv;
  cv.push_back(ControlVariate(2, {0, 0}));
  cv.push_back(ControlVariate(6, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  cv.push_back(ControlVariate(1, {0}));

  std::ofstream out_mp[3];
  Timer mcTimer, stepTimer;

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();

    for (int i = 0; i < 3; i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".2" + std::to_string(i + 2);
      out_mp[i].open(filename.c_str());
    }
  }

  // --- initialize
  update_wavefunction();
  tau.new_tau(random);
  ovps.update_ovps(el_pair_list.data(), tau);
  ovps.update_ovps_03(el_pair_list.data(), tau);

//  mcmp2_energy(emp[0], control[0]);
//  mcmp3_energy(emp[1], control[1]);
//  mcmp4_energy(emp[2], control[2]);

  // run monte carlo simulation
  for (int step = 1; step <= iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    for (auto& it : el_pair_list) {
      it.mc_move_scheme(random, molec, mc_basis);
    }

    // generate new tau values
    tau.new_tau(random);

    // update wavefunction and green's function traces
    update_wavefunction();
    ovps.update_ovps(el_pair_list.data(), tau);
    ovps.update_ovps_03(el_pair_list.data(), tau);

    // calcaulte energy for step
//    mcmp2_energy(emp[0], control[0]);
//    mcmp3_energy(emp[1], control[1]);
//    mcmp4_energy(emp[2], control[2]);

    // accumulate
    for (int i = 0; i < emp.size(); i++) {
      cv[i].add(emp[i], control[i]);
    }

    // print if i is a multiple of 128
    if (0 == step % 128) {
      for (int i = 0; i < emp.size(); i++) {
        cv[i].add(emp[i], control[i]);
        out_mp[i] << cv[i] << "\t";
        out_mp[i] << stepTimer << "\n";
        out_mp[i].flush();
      }
      stepTimer.Start();
    }
  }
  if (mpi_info.sys_master) {
    std::cout << "Spent " << mcTimer << " second preforming MC integration" << std::endl;
    for (int i = 0; i < emp.size(); i++) {
      out_mp[i].close();
    }
  }
}
 */

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

void MP3::monte_energy() {
  // variables to store emp2 energy
  double emp2, emp3;
  std::vector<double> mp2_control(2), mp3_control(6);
  ControlVariate cv_emp2(2, {0, 0}), cv_emp3(6, {0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0});
                                                 // 0.0, 0.0, 0.0,
                                                 // 0.0, 0.0, 0.0});

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
  tau.new_tau(random);
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

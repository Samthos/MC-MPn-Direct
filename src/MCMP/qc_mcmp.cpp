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

template <class Container>
void Energy<Container>::zero_energies() {
  std::fill(emp.begin(), emp.end(), 0.0);
  for (auto &c : control) {
    std::fill(c.begin(), c.end(), 0.0);
  }
}

template <class Container>
void Energy<Container>::monte_energy() {
  Timer mcTimer, stepTimer;
  std::vector<std::ofstream> output(emp.size()+1);

#ifdef DIMER_PRINT
  std::vector<std::ofstream> dimer_energy_output(emp.size() + 1);
  std::vector<std::ofstream> dimer_control_output(emp.size() + 1);
  MPI_info::broadcast_string(iops.sopns[KEYS::JOBNAME]);
#endif // DIMER_PRINT

  // open output stream and start clock for calculation
  if (mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();
    print_mc_head(mcTimer.StartTime());

    for (auto i = 0; i < energy_functions.size(); i++) {
      std::string filename = iops.sopns[KEYS::JOBNAME] + "." + energy_functions[i]->extension;
      output[i].open(filename);
    }
    {
      std::string filename = iops.sopns[KEYS::JOBNAME] + ".20";
      output.back().open(filename);
    }
  }

  // if DIMER_PRINT is defined: open binary ofstream for eneries and control
  // variates of each step, for each thread.
  // currently only works for MP2
#ifdef DIMER_PRINT
  for (auto i = 0; i < emp.size(); i++) {
    std::string dimer_filename = iops.sopns[KEYS::JOBNAME];
    dimer_filename += ".taskid_" + std::to_string(mpi_info.taskid) + ".";
    dimer_filename += energy_functions[i]->extension;

    dimer_energy_output[i].open(dimer_filename + ".emp.bin", std::ios::binary);
    dimer_control_output[i].open(dimer_filename + ".cv.bin", std::ios::binary);
  }
  {
    std::string dimer_filename = iops.sopns[KEYS::JOBNAME];
    dimer_filename += ".taskid_" + std::to_string(mpi_info.taskid) + ".20";

    dimer_energy_output.back().open(dimer_filename + ".emp.bin", std::ios::binary);
    dimer_control_output.back().open(dimer_filename + ".cv.bin", std::ios::binary);
  }
#endif // DIMER_PRINT

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
    
// dump energies and control vars to dimer binary ofstream
#ifdef DIMER_PRINT
    for (auto it = 0; it < cv.size()-1; it++) {
      dimer_energy_output[it].write((char*) &emp[it], sizeof(double));
      dimer_control_output[it].write((char*) control[it].data(), sizeof(double) * control[it].size());
    }
    {
      double total_energy = std::accumulate(emp.begin(), emp.end(), 0.0);
      dimer_energy_output.back().write((char*) &total_energy, sizeof(double));
      dimer_control_output.back().write((char*) control.back().data(), sizeof(double) * control.back().size());
    }
#endif // DIMER_PRINT

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

template <class Container>
void Energy<Container>::energy() {
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

#ifdef HAVE_CUDA
void GPU_Energy::energy() {
  ovps_device.update_ovps(wavefunctions[WC::electron_pairs_1], wavefunctions[WC::electron_pairs_2], tau);
  copy_OVPS(ovps_device, ovps);
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
#endif

template <class Binary_Op>
void Dimer::local_energy(std::unordered_map<int, Wavefunction>& l_wavefunctions, Tau* l_tau, Binary_Op op) {
  std::fill(l_emp.begin(), l_emp.end(), 0.0);
  for (auto &c : l_control) {
    std::fill(c.begin(), c.end(), 0.0);
  }

  ovps.update_ovps(l_wavefunctions[WC::electron_pairs_1], l_wavefunctions[WC::electron_pairs_2], l_tau);
  for (int i = 0; i < energy_functions.size(); i++) {
    if (!energy_functions[i]->is_f12) {
      if (l_tau->is_new(energy_functions[i]->n_tau_coordinates)) {
        energy_functions[i]->energy(l_emp[i], l_control[i], ovps, electron_pair_list, l_tau);
      }
    } else {
      if (l_tau->is_new(energy_functions[i]->n_tau_coordinates)) {
        energy_functions[i]->energy_f12(l_emp[i], l_control[i], l_wavefunctions, electron_pair_list, electron_list);
      }
    }
  }

  std::transform(emp.begin(), emp.end(), l_emp.begin(), emp.begin(), op);
  for (auto it = control.begin(), jt = l_control.begin(); it != control.end(); it++, jt++) {
    std::transform(it->begin(), it->end(), jt->begin(), it->begin(), op);
  }
}

void Dimer::energy() {
  monomer_a_tau->set_from_other(tau);
  monomer_b_tau->set_from_other(tau);

  local_energy(monomer_a_wavefunctions, monomer_a_tau, std::minus<>());
  local_energy(monomer_b_wavefunctions, monomer_b_tau, std::minus<>());
  local_energy(wavefunctions, tau, std::plus<>());
}

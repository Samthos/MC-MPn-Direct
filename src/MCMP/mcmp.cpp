//
// Created by aedoran on 6/1/18.
//
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../MCF12/mp2_f12.h"
#include "create_mp2_functional.h"
#include "mp3_functional.h"
#include "mp4_functional.h"

#include "../qc_monte.h"
#include "../control_variate.h"
#include "../timer.h"

#include "mcmp.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
MCMP<Container, Allocator>::MCMP(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : QC_monte<Container, Allocator>(p1, p2, p3, p4) {
  int max_tau_coordinates = 0;
  int total_control_variates = 0;

  if (this->iops.iopns[KEYS::TASK] & TASK::MP2) {
    energy_functions.push_back(create_MP2_Functional<Container, Allocator>(this->iops.iopns[KEYS::MP2CV_LEVEL], this->iops.iopns[KEYS::ELECTRON_PAIRS]));
  }
  if (this->iops.iopns[KEYS::TASK] & TASK::DIRECT_MP2) {
    energy_functions.push_back(create_Direct_MP2_Functional(this->iops.iopns[KEYS::MP2CV_LEVEL]));
  }
  if (this->iops.iopns[KEYS::TASK] & TASK::MP3) {
    energy_functions.push_back(create_MP3_Functional(this->iops.iopns[KEYS::MP3CV_LEVEL], this->electron_pair_list->size()));
  }
  if (this->iops.iopns[KEYS::TASK] & TASK::MP4) {
    energy_functions.push_back(create_MP4_Functional(this->iops.iopns[KEYS::MP4CV_LEVEL], this->electron_pair_list->size()));
  }
  if (this->iops.iopns[KEYS::TASK] & TASK::MP2_F12_V) {
    energy_functions.push_back(new MP2_F12_V(this->iops));
  }
  if (this->iops.iopns[KEYS::TASK] & TASK::MP2_F12_VBX) {
    energy_functions.push_back(new MP2_F12_VBX(this->iops));
  }

  emp.resize(energy_functions.size());
  for (auto &it : energy_functions) {
    control.emplace_back(it->n_control_variates);
    cv.push_back(create_accumulator(this->electron_pair_list->requires_blocking(), std::vector<double>(it->n_control_variates, 0.0)));

    max_tau_coordinates = std::max(max_tau_coordinates, it->n_tau_coordinates);
    total_control_variates += it->n_control_variates;
  }

  this->tau->resize(max_tau_coordinates);
  this->ovps.init(max_tau_coordinates, this->iops.iopns[KEYS::ELECTRON_PAIRS]);
  
  if (energy_functions.size() == 1) {
    Direct_MP_Functional_Type* functional = dynamic_cast<Direct_MP_Functional_Type*>(energy_functions[0]);
    if (functional) {
      this->ovps.init(0, this->iops.iopns[KEYS::ELECTRON_PAIRS]);
    }
  } 
  
  control.emplace_back(total_control_variates);
  cv.push_back(create_accumulator(this->electron_pair_list->requires_blocking(), std::vector<double>(total_control_variates, 0.0)));
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
MCMP<Container, Allocator>::~MCMP() {
  for (auto &item : cv) {
    delete item;
  }
  for (auto &item : energy_functions) {
    delete item;
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MCMP<Container, Allocator>::zero_energies() {
  std::fill(emp.begin(), emp.end(), 0.0);
  for (auto &c : control) {
    std::fill(c.begin(), c.end(), 0.0);
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MCMP<Container, Allocator>::monte_energy() {
  Timer mcTimer, stepTimer;
  std::vector<std::ofstream> output(emp.size()+1);

#ifdef DIMER_PRINT
  std::vector<std::ofstream> dimer_energy_output(emp.size() + 1);
  std::vector<std::ofstream> dimer_control_output(emp.size() + 1);
  MPI_info::broadcast_string(iops.sopns[KEYS::JOBNAME]);
#endif // DIMER_PRINT

  // open output stream and start clock for calculation
  if (this->mpi_info.sys_master) {
    mcTimer.Start();
    stepTimer.Start();
    this->print_mc_head(mcTimer.StartTime());

    for (auto i = 0; i < energy_functions.size(); i++) {
      std::string filename = this->iops.sopns[KEYS::JOBNAME] + "." + energy_functions[i]->extension;
      output[i].open(filename);
    }
    {
      std::string filename = this->iops.sopns[KEYS::JOBNAME] + ".20";
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
  for (int step = 1; step <= this->iops.iopns[KEYS::MC_TRIAL]; step++) {
    // generate new positions
    this->move_walkers();

    // update wavefunction
    this->update_wavefunction();

    // zero energy arrarys
    zero_energies();

    // calcaulte energy for step
    do {
      this->tau->new_tau(this->random);
      energy();
    } while (this->tau->next());

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

      for (auto i = 0; i < emp.size(); i++) {
          std::string filename = this->iops.sopns[KEYS::JOBNAME] + "." + energy_functions[i]->extension;
          cv[i]->to_json(filename);
      }
      {
        std::string filename = this->iops.sopns[KEYS::JOBNAME] + ".20";
        cv.back()->to_json(filename);
      }
    }
  }


  if (this->mpi_info.sys_master) {
    mcTimer.Stop();
    this->print_mc_tail(mcTimer.Span(), mcTimer.EndTime());
    for (auto i = 0; i < emp.size(); i++) {
      output[i].close();
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void MCMP<Container, Allocator>::energy() {
  this->ovps.update(this->wavefunctions[WC::electron_pairs_1], this->wavefunctions[WC::electron_pairs_2], this->tau);
  for (int i = 0; i < energy_functions.size(); i++) {
    if (this->tau->is_new(energy_functions[i]->n_tau_coordinates)) {
      if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::STANDARD) {
        Standard_MP_Functional_Type* functional = dynamic_cast<Standard_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(emp[i], control[i], this->ovps, this->electron_pair_list, this->tau);
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::F12) {
        F12_MP_Functional_Type* functional = dynamic_cast<F12_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(emp[i], control[i], this->wavefunctions, this->electron_pair_list, this->electron_list); 
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::DIRECT) {
        Direct_MP_Functional_Type* functional = dynamic_cast<Direct_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(emp[i], control[i], this->wavefunctions[WC::electron_pairs_1], this->wavefunctions[WC::electron_pairs_2], this->electron_pair_list, this->tau); 
      }
    }
  }
}


#ifdef HAVE_CUDA
template <> void MCMP<thrust::device_vector, thrust::device_allocator>::energy() {}

GPU_MCMP::GPU_MCMP(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : MCMP(p1, p2, p3, p4) {
}
 
void GPU_MCMP::energy() {
  this->ovps.update(this->wavefunctions[WC::electron_pairs_1], this->wavefunctions[WC::electron_pairs_2], this->tau);
  for (int i = 0; i < energy_functions.size(); i++) {
    if (this->tau->is_new(energy_functions[i]->n_tau_coordinates)) {
      if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::STANDARD) {
        Standard_MP_Functional_Type* functional = dynamic_cast<Standard_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(emp[i], control[i], this->ovps, this->electron_pair_list, this->tau);
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::F12) {
        // F12_MP_Functional* functional = dynamic_cast<F12_MP_Functional*>(energy_functions[i]);
        // functional->energy(emp[i], control[i], this->wavefunctions, this->electron_pair_list, this->electron_list); 
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::DIRECT) {
        // Direct_MP_Functional* functional = dynamic_cast<Direct_MP_Functional*>(energy_functions[i]);
        // functional->energy(emp[i], control[i], this->wavefunctions[WC::electron_pairs_1], this->wavefunctions[WC::electron_pairs_2], this->electron_pair_list, this->tau); 
      }
    }
  }
}
#endif


Dimer::Dimer(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : MCMP(p1, p2, p3, p4),
                                                         l_emp(emp),
                                                         l_control(control)
{
  Molecule monomer_a_geometry(mpi_info, iops.sopns[KEYS::MONOMER_A_GEOM]);
  Molecule monomer_b_geometry(mpi_info, iops.sopns[KEYS::MONOMER_B_GEOM]);

  auto monomer_a_movecs = create_movec_parser(mpi_info, monomer_a_geometry, MOVEC_TYPE::NWCHEM_BINARY, iops.sopns[KEYS::MONOMER_A_MOVECS], iops.bopns[KEYS::FREEZE_CORE]);
  auto monomer_b_movecs = create_movec_parser(mpi_info, monomer_b_geometry, MOVEC_TYPE::NWCHEM_BINARY, iops.sopns[KEYS::MONOMER_B_MOVECS], iops.bopns[KEYS::FREEZE_CORE]);

  monomer_a_tau = create_tau_sampler(static_cast<TAU_GENERATORS::TAU_GENERATORS>(iops.iopns[KEYS::TAU_GENERATORS]), monomer_a_movecs);
  monomer_b_tau = create_tau_sampler(static_cast<TAU_GENERATORS::TAU_GENERATORS>(iops.iopns[KEYS::TAU_GENERATORS]), monomer_b_movecs);

  monomer_a_tau->resize(tau->get_n_coordinates());
  monomer_b_tau->resize(tau->get_n_coordinates());

  for (auto &it : wavefunctions) {
    auto pos_source = it.second.pos;
    monomer_a_wavefunctions.emplace(it.first, Wavefunction_Type(pos_source, monomer_a_movecs));
    monomer_b_wavefunctions.emplace(it.first, Wavefunction_Type(pos_source, monomer_b_movecs));
  }
}

Dimer::~Dimer() {
}

/*
Dimer::Dimer(MPI_info p1, IOPs p2, Molecule p3, Basis p4) : QC_monte(p1, p2, p3, p4) {
  // build temp list of current wavefunctions keys

  // loop over wavefunctions keys.
  //     wavefunctions.emplace(monomer_a & key, wavefunction(monomer_a))
  //     wavefunctions.emplace(monomer_b & key, wavefunction(monomer_b))
  //     wavefunction_groups[key & mask].push_back(monomer_a & key)
  //     wavefunction_groups[key & mask].push_back(monomer_b & key)

  // ^^^^need to update WT::mask with new dimer/monomer keys

  // create extra tau functions
}
*/

void Dimer::update_wavefunction() {
  for (auto &it : wavefunction_groups) {
    if (it.second.size() == 1) {
      basis.build_contractions(*wavefunctions[it.second.front()].pos);
    } else {
      basis.build_contractions_with_derivatives(*wavefunctions[it.second.front()].pos);
    }
    for (auto &jt : it.second) {
      Wavefunction_Type& wavefunction = wavefunctions[jt];
      Wavefunction_Type& monomer_a_wavefunction = monomer_a_wavefunctions[jt];
      Wavefunction_Type& monomer_b_wavefunction = monomer_b_wavefunctions[jt];
      auto code = jt & WT::mask;
      switch (code) {
        case WT::normal: basis.build_ao_amplitudes(*wavefunction.pos); break;
        case WT::dx:     basis.build_ao_amplitudes_dx(*wavefunction.pos); break;
        case WT::dy:     basis.build_ao_amplitudes_dy(*wavefunction.pos); break;
        case WT::dz:     basis.build_ao_amplitudes_dz(*wavefunction.pos); break;
      }
      wavefunction.ao_to_mo(basis.ao_amplitudes);
      monomer_a_wavefunction.ao_to_mo(basis.ao_amplitudes);
      monomer_b_wavefunction.ao_to_mo(basis.ao_amplitudes);
    }
  }
}

template <class Binary_Op>
void Dimer::local_energy(std::unordered_map<int, Wavefunction_Type>& l_wavefunctions, Tau* l_tau, Binary_Op op) {
  std::fill(l_emp.begin(), l_emp.end(), 0.0);
  for (auto &c : l_control) {
    std::fill(c.begin(), c.end(), 0.0);
  }

  ovps.update(l_wavefunctions[WC::electron_pairs_1], l_wavefunctions[WC::electron_pairs_2], l_tau);
  for (int i = 0; i < energy_functions.size(); i++) {
    if (tau->is_new(energy_functions[i]->n_tau_coordinates)) {
      if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::STANDARD) {
        Standard_MP_Functional_Type* functional = dynamic_cast<Standard_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(l_emp[i], l_control[i], ovps, electron_pair_list, l_tau);
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::F12) {
        F12_MP_Functional_Type* functional = dynamic_cast<F12_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(l_emp[i], l_control[i], l_wavefunctions, electron_pair_list, electron_list); 
      } else if (energy_functions[i]->functional_type == MP_FUNCTIONAL_TYPE::DIRECT) {
        Direct_MP_Functional_Type* functional = dynamic_cast<Direct_MP_Functional_Type*>(energy_functions[i]);
        functional->energy(l_emp[i], l_control[i], l_wavefunctions[WC::electron_pairs_1], l_wavefunctions[WC::electron_pairs_2], electron_pair_list, l_tau); 
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

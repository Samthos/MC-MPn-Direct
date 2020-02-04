// Copyright 2017

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


#include "qc_mpi.h"
#include "qc_input.h"
#include "MCF12/correlation_factors.h"

IOPs::IOPs() {
  /*
   * IOPs constructor
   * initializes variables to mostly rational values
   *
   * TODO
   *  -read values directly instead of setting them
   */
  bopns[KEYS::SPHERICAL] = true;
  bopns[KEYS::F12_GAMMA] = false;
  bopns[KEYS::F12_BETA] = false;

  dopns[KEYS::MC_DELX] = 0.1;

  iopns[KEYS::MC_NPAIR] = 16;
  iopns[KEYS::ELECTRON_PAIRS] = 16;
  iopns[KEYS::ELECTRONS] = 16;
  iopns[KEYS::MC_TRIAL] = 1024;
  iopns[KEYS::MC_PAIR_GROUPS] = 1;
  iopns[KEYS::OFF_BAND] = 1;
  iopns[KEYS::NUM_BAND] = 1;
  iopns[KEYS::DIFFS] = 1;
  iopns[KEYS::NBLOCK] = 1;
  iopns[KEYS::ORDER] = 2;
  iopns[KEYS::TASK] = TASKS::MP;
  iopns[KEYS::CPU] = true;
  iopns[KEYS::SAMPLER] = SAMPLERS::DIRECT;
  iopns[KEYS::TAU_INTEGRATION] = TAU_INTEGRATION::STOCHASTIC;
  iopns[KEYS::F12_CORRELATION_FACTOR] = CORRELATION_FACTORS::Slater;

  sopns[KEYS::GEOM] = "geom.xyz";
  sopns[KEYS::BASIS] = "basis.dat";
  sopns[KEYS::MC_BASIS] = "mc_basis.dat";
  sopns[KEYS::MOVECS] = "nwchem.movecs";
  iopns[KEYS::MOVECS] = 0;  // default is binary
}

void IOPs::read(const MPI_info& mpi_info, const std::string& file) {
  /*
   * reads and stores options mcin file provided as command line argument
   *
   * Arguments
   *  -MPI_info mpi_info: see qc_mpi.h
   *  -std::string file: path to mcin file
   *
   * TODO
   *  -should probably read input from a json
   *  -needs input validation
   *  -make key setter by type
   *  -write functions to convert strings to enums
   *  -clean up keys
   */
  KEYS::KeyVal keyval;

  bool keySet;
  std::string str;
  std::string key;

  const std::vector<std::string> key_vals = {
      "JOBNAME", "SPHERICAL", "MC_TRIAL", "MC_NPAIR", "MC_DELX",  //  0-4
      "GEOM", "BASIS", "MC_BASIS", "NBLOCK", "MOVECS",            //  5-9
      "DEBUG", "MC_PAIR_GROUPS", "TASK", "NUM_BAND", "OFF_BAND",  // 10-14
      "DIFFS", "ORDER", "CPU", "SAMPLER", "TAU_INTEGRATION",
      "F12_CORRELATION_FACTOR", "F12_GAMMA", "F12_BETA", "ELECTRONS", "ELECTRON_PAIRS"
  };
  const std::vector<std::string> taskVals = {
      "MP", "GF", "GFDIFF", "GFFULL", "GFFULLDIFF", "F12V"};

  if (mpi_info.sys_master) {
    std::ifstream input(file.c_str());
    if (!input.is_open()) {
      std::cerr << "FILE \"" << file << "\" DOES NOT EXIST" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    keySet = false;
    while (getline(input, str)) {
      std::istringstream ss(str);
      while (ss >> key) {
        if (keySet == false) {  // if key not set, determine key value from key_vals arrays
          std::transform(key.begin(), key.end(), key.begin(), ::toupper);
          auto it = std::find(key_vals.begin(), key_vals.end(), key);
          if (it != key_vals.end()) {
            keyval = static_cast<KEYS::KeyVal>(std::distance(key_vals.begin(), it));
          } else {
            std::cerr << "Key " << key << " not reconginzed" << std::endl;
            exit(0);
          }
          keySet = true;
        } else {
          switch (keyval) {
            case KEYS::TASK:
              std::transform(key.begin(), key.end(), key.begin(), ::toupper);
              {
                auto it = std::find(taskVals.begin(), taskVals.end(), key);
                if (it != taskVals.end()) {
                  iopns[keyval] = std::distance(taskVals.begin(), it);
                } else {
                  std::cerr << "Task " << key << " not reconginzed" << std::endl;
                  exit(0);
                }
              }
              keySet = false;
              break;
            case KEYS::JOBNAME:
              sopns[keyval] = key;
              keySet = false;
              break;
            case KEYS::SPHERICAL:
              bopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::MC_TRIAL:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::DIFFS:
              iopns[keyval] = stoi(key, nullptr) + 1;
              if (iopns[keyval] <= 0) {
                iopns[keyval] = 1;
              }
              keySet = false;
              break;
            case KEYS::OFF_BAND:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::NUM_BAND:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::ELECTRON_PAIRS:
              keyval = KEYS::MC_NPAIR;
            case KEYS::MC_NPAIR:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::ELECTRONS:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::MC_DELX:
              dopns[keyval] = stod(key, nullptr);
              keySet = false;
              break;
            case KEYS::GEOM:
              sopns[keyval] = key;
              keySet = false;
              break;
            case KEYS::ORDER:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::BASIS:
              sopns[keyval] = key;
              keySet = false;
              break;
            case KEYS::MC_BASIS:
              sopns[keyval] = key;
              keySet = false;
              break;
            case KEYS::NBLOCK:
              iopns[keyval] = stoi(key, nullptr);
              if (iopns[keyval] > 20) {
                std::cerr << "NBlock must be less than 20" << std::endl;
                exit(EXIT_FAILURE);
              } else if (iopns[keyval] <= 0) {
                iopns[keyval] = 1;
              }
              keySet = false;
              break;
            case KEYS::MOVECS:
              if (key == "ASCII") {
                iopns[keyval] = 1;
              } else if (key == "END") {
                keySet = false;
              } else {
                sopns[keyval] = key;
              }
              break;
            case KEYS::SAMPLER:
              if (key == "DIRECT") {
                iopns[keyval] = SAMPLERS::DIRECT;
              } else if (key == "METROPOLIS") {
                iopns[keyval] = SAMPLERS::METROPOLIS;
              } else {
                std::cerr << key << " is not a vaild sampler\n";
              }
              keySet = false;
              break;
            case KEYS::TAU_INTEGRATION:
              if (key == "STOCHASTIC") {
                iopns[keyval] = TAU_INTEGRATION::STOCHASTIC;
              } else if (key == "SUPER_STOCH") {
                iopns[keyval] = TAU_INTEGRATION::SUPER_STOCH;
              } else if (key == "QUADRATURE") {
                iopns[keyval] = TAU_INTEGRATION::QUADRATURE;
              } else {
                std::cerr << key << " is not a vaild tau integration method\n";
              }
              keySet = false;
              break;
            case KEYS::DEBUG:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::MC_PAIR_GROUPS:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::CPU:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::F12_CORRELATION_FACTOR:
              iopns[keyval] = string_to_correlation_factors(key);
              keySet = false;
              break;
            case KEYS::F12_GAMMA:
              bopns[keyval] = true;
              dopns[keyval] = stod(key, nullptr);
              keySet = false;
              break;
            case KEYS::F12_BETA:
              bopns[keyval] = true;
              dopns[keyval] = stod(key, nullptr);
              keySet = false;
              break;
            default:
              std::cerr << "KEY \"" << key << "\" NOT RECONGNIZED" << std::endl;
              exit(EXIT_FAILURE);
          }
        }
      }
    }
    input.close();
  }

  if (iopns[KEYS::TASK] == TASKS::GF || iopns[KEYS::TASK] == TASKS::GFFULL) {
    iopns[KEYS::DIFFS] = 1;
  }

  MPI_info::barrier();
  MPI_info::broadcast_int(iopns.data(), iopns.size());
  MPI_info::broadcast_double(dopns.data(), dopns.size());
  MPI_info::broadcast_char((char*) bopns.data(), bopns.size());
}

void IOPs::print(const MPI_info& mpi_info, const std::string& file) {
  /*
   * prints state of iops
   *
   * Arguments
   *  -MPI_info mpi_info: see qc_mpi.h
   *  -std::string file: path to mcin file used to set iops values
   *
   * TODO
   *  -should probably read input from a json
   */
  const std::vector<std::string> taskVals = {
      "MP", "GF", "GFDIFF", "GFFULL", "GFFULLDIFF", "F12V"};
  const std::vector<std::string> samplers = {
      "DIRECT", "METROPOLIS"};
  const std::vector<std::string> tau_integrations = {
      "STOCHASTIC", "QUADRATURE", "SUPER_STOCH"};

  if (mpi_info.sys_master) {
    std::cout << std::endl;
    std::cout << "Input read from " << file << std::endl;
    std::cout << "JOBNAME: " << sopns[KEYS::JOBNAME] << std::endl;
    std::cout << " TASK: " << taskVals[iopns[KEYS::TASK]] << std::endl;
    std::cout << " ORDER: " << iopns[KEYS::ORDER] << std::endl;
    std::cout << " MC_TRIAL: " << iopns[KEYS::MC_TRIAL] << std::endl;
    std::cout << " MC_NPAIR: " << iopns[KEYS::MC_NPAIR] << std::endl;
    std::cout << " MC_DELX: " << dopns[KEYS::MC_DELX] << std::endl;
    std::cout << " SPHERICAL: " << bopns[KEYS::SPHERICAL] << std::endl;
    std::cout << " SAMPLER: " << samplers[iopns[KEYS::SAMPLER]] << std::endl;
    std::cout << " TAU_INTEGRATION: " << tau_integrations[iopns[KEYS::TAU_INTEGRATION]]  << std::endl;
    if (iopns[KEYS::DEBUG] == 1) {
      std::cout << " RNG in debug mode" << std::endl;
    }

    if (iopns[KEYS::TASK] == TASKS::GF || iopns[KEYS::TASK] == TASKS::GFDIFF || iopns[KEYS::TASK] == TASKS::GFFULL || iopns[KEYS::TASK] == TASKS::GFFULLDIFF) {
      std::cout << "\tDIFFS = ";
      if (iopns[KEYS::DIFFS] > 1) {
        std::cout << iopns[KEYS::DIFFS] - 1 << std::endl;
      } else {
        std::cout << "FALSE" << std::endl;
      }
      std::cout << "\tBands: ";

      for (int i = 0; i < iopns[KEYS::NUM_BAND]; i++) {
        if ((i + 1 - iopns[KEYS::OFF_BAND]) < 0) {
          std::cout << "HOMO" << i - iopns[KEYS::OFF_BAND] + 1;
        } else if ((i + 1 - iopns[KEYS::OFF_BAND]) == 0) {
          std::cout << "HOMO-" << i - iopns[KEYS::OFF_BAND] + 1;
        } else {
          std::cout << "LUMO+" << i - iopns[KEYS::OFF_BAND];
        }

        if (i < iopns[KEYS::NUM_BAND] - 1) {
          std::cout << ", ";
        }
      }
    } else if (iopns[KEYS::TASK] == TASKS::F12V) {
      std::cout << "Number of Electrons Walkers = " << iopns[KEYS::ELECTRONS] << "\n";
      std::cout << "Correlation Factor = " << correlation_factors_to_string(static_cast<CORRELATION_FACTORS::CORRELATION_FACTORS>(iopns[KEYS::F12_CORRELATION_FACTOR])) << "\n";

      std::cout << "F12_GAMMA = ";
      if (bopns[KEYS::F12_GAMMA]) {
        std::cout << "default";
      } else {
        std::cout << dopns[KEYS::F12_GAMMA];
      }
      std::cout << "\n";

      std::cout << "F12_BETA = ";
      if (bopns[KEYS::F12_BETA]) {
        std::cout << "default";
      } else {
        std::cout << dopns[KEYS::F12_BETA];
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }
}

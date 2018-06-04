// Copyright 2017

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif  // HAVE_CONFIG_H

#ifdef USE_MPI
#include "mpi.h"
#endif
#include "qc_input.h"

IOPs::IOPs() {
  bopns[KEYS::SPHERICAL] = true;
  dopns[KEYS::MC_DELX] = 0.1;
  iopns[KEYS::MC_NPAIR] = 16;
  iopns[KEYS::MC_TRIAL] = 1024;
  iopns[KEYS::MC_PAIR_GROUPS] = 1;
  iopns[KEYS::OFF_BAND] = 1;
  iopns[KEYS::NUM_BAND] = 1;
  iopns[KEYS::DIFFS] = 1;
  iopns[KEYS::NBLOCK] = 1;
  iopns[KEYS::ORDER] = 2;
  iopns[KEYS::TASK] = TASKS::MP;

  sopns[KEYS::GEOM] = "geom.xyz";
  sopns[KEYS::BASIS] = "basis.dat";
  sopns[KEYS::MC_BASIS] = "mc_basis.dat";
  sopns[KEYS::MOVECS] = "nwchem.movecs";
  iopns[KEYS::MOVECS] = 0;  // default is binary
}

void IOPs::read(const MPI_info& mpi_info,
                const std::string& file) {
  KEYS::KeyVal keyval;

  bool keySet;
  std::string str;
  std::string key;

  const std::vector<std::string> key_vals = {
      "JOBNAME", "SPHERICAL", "MC_TRIAL", "MC_NPAIR", "MC_DELX",  //  0-4
      "GEOM", "BASIS", "MC_BASIS", "NBLOCK", "MOVECS",            //  5-9
      "DEBUG", "MC_PAIR_GROUPS", "TASK", "NUM_BAND", "OFF_BAND",  // 10-14
      "DIFFS", "ORDER"};
  const std::vector<std::string> taskVals = {
      "MP", "GF", "GFDIFF", "GFFULL", "GFFULLDIFF"};

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
          /*
          std::transform(key.begin(), key.end(), key.begin(), ::toupper);
          for (i=0;i<key_vals.size();i++)
          {
            if (key_vals[i] == key)
            {
              keyval = static_cast<KEYS::KeyVal>(i);
              break;
            }
          }
          keySet = true;
          */
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
              job_name = key;
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
            case KEYS::MC_NPAIR:
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
            case KEYS::DEBUG:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            case KEYS::MC_PAIR_GROUPS:
              iopns[keyval] = stoi(key, nullptr);
              keySet = false;
              break;
            default:
              std::cerr << "KEY \"" << key << "\" NOT RECONGNIZED" << std::endl;
              exit(EXIT_FAILURE);
              break;
          }
        }
      }
    }
    input.close();
  }

  if (iopns[KEYS::TASK] == TASKS::GF || iopns[KEYS::TASK] == TASKS::GFFULL) {
    iopns[KEYS::DIFFS] = 1;
  }

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(iopns.data(), iopns.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(dopns.data(), dopns.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(bopns.data(), bopns.size(), MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void IOPs::print(const MPI_info& mpi_info,
                 const std::string& file) {
  const std::vector<std::string> taskVals = {
      "MP", "GF", "GFDIFF", "GFFULL", "GFFULLDIFF"};

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
#ifdef QUAD_TAU
    std::cout << " quadrature tau" << std::endl;
#else
    std::cout << " Stocahstic tau" << std::endl;
#endif
    if (iopns[KEYS::DEBUG] == 1) {
      std::cout << " RNG in debug mode" << std::endl;
    }

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
    std::cout << std::endl;
  }
}

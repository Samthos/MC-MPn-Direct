// Copyright 2017
#ifndef QC_INPUT_H_
#define QC_INPUT_H_

#include <array>
#include <string>
#include <vector>

#include "qc_mpi.h"

namespace KEYS {
  enum KEYS {
    JOBNAME = 0,             // simple 
    JOBTYPE,                 // simple
    SPHERICAL,               // basis
    MC_TRIAL,                // simple
    MC_DELX,                 // sampler?
    GEOM,                    // geometry
    BASIS,                   // basis
    MC_BASIS,                // mc basis / sampler
    NBLOCK,                  // try to depreciate
    MOVECS,                  // basis?
    DEBUG,                   // simple
    TASK,                    // ????
    NUM_BAND,                // gf
    OFF_BAND,                // gf
    DIFFS,                   // gf
    ORDER,                   // ????????
    SAMPLER,                 // sampler
    TAU_INTEGRATION,         // sampler
    F12_CORRELATION_FACTOR,  // F12
    F12_GAMMA,               // F12
    F12_BETA,                // F12
    ELECTRONS,               // simple
    ELECTRON_PAIRS,          // simple
    MP2CV_LEVEL,
    MP3CV_LEVEL,
    MP4CV_LEVEL,
  };
  const std::vector<std::string> key_strings = {
      "JOBNAME", "JOBTYPE", "SPHERICAL", "MC_TRIAL", "MC_DELX",
      "GEOM", "BASIS", "MC_BASIS", "NBLOCK", "MOVECS",
      "DEBUG", "TASK", "NUM_BAND", "OFF_BAND", 
      "DIFFS", "ORDER", "SAMPLER", "TAU_INTEGRATION",
      "F12_CORRELATION_FACTOR", "F12_GAMMA", "F12_BETA", "ELECTRONS", "ELECTRON_PAIRS",
      "MP2CV_LEVEL", "MP3CV_LEVEL", "MP4CV_LEVEL",
  };
}
namespace SAMPLER {
  enum SAMPLER {
    DIRECT,
    METROPOLIS,
  };
  const std::vector<std::string> sampler_strings = {
    "DIRECT",
    "METROPOLIS"
  };
}
namespace TAU_INTEGRATION {
  enum TAU_INTEGRATION {
    STOCHASTIC,
    QUADRATURE,
    SUPER_STOCH,
  };
  const std::vector<std::string> tau_integration_strings = {
    "STOCHASTIC",
    "QUADRATURE",
    "SUPER_STOCH"
  };
}
namespace JOBTYPE {
  enum JOBTYPE {
    ENERGY,
    DIMER_ENERGY,
    GF,
    GFDIFF,
    GFFULL,
    GFFULLDIFF,
    MP,
    F12V,
    F12VBX,
  };
  const std::vector<std::string> jobtype_strings = {
    "ENERGY",
    "DIMER_ENERGY",
    "GF",
    "GFDIFF",
    "GFFULL",
    "GFFULLDIFF",
    "MP",
    "F12V",
    "F12VBX"
  };
}
namespace TASK {
  enum TASK {
    MP2 =         0b0000001,
    MP3 =         0b0000010,
    MP4 =         0b0000100,
    MP2_F12_V =   0b0001000,
    MP2_F12_VBX = 0b0010000,
    GF2 =         0b0100000,
    GF3 =         0b1000000,
  };
  const std::vector<std::string> task_strings = {
    "MP2", 
    "MP3", 
    "MP4",
    "MP2_F12_V",
    "MP2_F12_VBX", 
    "GF2", 
    "GF3"
  };
}

class IOPs {
 public:
  IOPs();

  void print(const MPI_info&, const std::string&);
  void read(const MPI_info&, const std::string&);

  std::array<int, 100> iopns;
  std::array<double, 100> dopns;
  std::array<bool, 100> bopns;
  std::array<std::string, 100> sopns;
};
#endif  // QC_INPUT_H_

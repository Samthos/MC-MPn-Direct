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
    ORDER,                   // depreciate
    SAMPLER,                 // sampler
    TAU_INTEGRATION,         // sampler
    F12_CORRELATION_FACTOR,  // F12
    F12_GAMMA,               // F12
    F12_BETA,                // F12
    ELECTRONS,               // simple
    ELECTRON_PAIRS,          // simple
    MP2CV_LEVEL,             // simple
    MP3CV_LEVEL,             // simple
    MP4CV_LEVEL,             // simple
    FREEZE_CORE,             // simple
    SEED_FILE,
  };
  const std::vector<std::string> key_strings = {
      "JOBNAME", "JOBTYPE", "SPHERICAL", "MC_TRIAL", "MC_DELX",
      "GEOM", "BASIS", "MC_BASIS", "NBLOCK", "MOVECS",
      "DEBUG", "TASK", "NUM_BAND", "OFF_BAND", 
      "DIFFS", "ORDER", "SAMPLER", "TAU_INTEGRATION",
      "F12_CORRELATION_FACTOR", "F12_GAMMA", "F12_BETA", "ELECTRONS", "ELECTRON_PAIRS",
      "MP2CV_LEVEL", "MP3CV_LEVEL", "MP4CV_LEVEL", "FREEZE_CORE", "SEED_FILE"
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
  };
  const std::vector<std::string> jobtype_strings = {
    "ENERGY",
    "DIMER_ENERGY",
    "GF",
    "GFDIFF",
    "GFFULL",
    "GFFULLDIFF",
  };
}
namespace TASK {
  enum TASK {
    MP2 =         0b000000001,
    MP3 =         0b000000010,
    MP4 =         0b000000100,
    MP2_F12_V =   0b000001000,
    MP2_F12_VBX = 0b000010000,
    GF2 =         0b000100000,
    GF3 =         0b001000000,
    GF2_F12_V =   0b010000000,
    GF2_F12_VBX = 0b100000000,
    ANY_F12     = 0b110011000,
    ANY_F12_VBX = 0b100010000,
  };
  const std::vector<std::string> task_strings = {
    "MP2", 
    "MP3", 
    "MP4",
    "MP2_F12_V",
    "MP2_F12_VBX", 
    "GF2", 
    "GF3",
    "GF2_F12_V",
    "GF2_F12_VBX", 
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

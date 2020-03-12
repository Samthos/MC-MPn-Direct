// Copyright 2017
#ifndef QC_INPUT_H_
#define QC_INPUT_H_

#include <array>
#include <string>
#include <vector>

#include "qc_mpi.h"

namespace KEYS {
  enum KeyVal {
    JOBNAME = 0,
    SPHERICAL,
    MC_TRIAL,
    MC_NPAIR,
    MC_DELX,  // 0-4
    GEOM,
    BASIS,
    MC_BASIS,
    NBLOCK,
    MOVECS,  // 5-9
    DEBUG,
    MC_PAIR_GROUPS,
    TASK,
    NUM_BAND,
    OFF_BAND,
    DIFFS,
    ORDER,
    CPU,
    SAMPLER,
    TAU_INTEGRATION,
    F12_CORRELATION_FACTOR,
    F12_GAMMA,
    F12_BETA,
    ELECTRONS,
    ELECTRON_PAIRS
  };
}
namespace SAMPLERS {
  enum SAMPLERS {
    DIRECT,
    METROPOLIS
  };
}
namespace TAU_INTEGRATION {
  enum METHODS {
    STOCHASTIC,
    QUADRATURE,
    SUPER_STOCH
  };
}
namespace TASKS {
  enum TaskVal {
    MP = 0,
    GF,
    GFDIFF,
    GFFULL,
    GFFULLDIFF,
    F12V,
    F12VBX,
    ENERGY
  };
  const std::vector<std::string> taskVals = {
      "MP", "GF", "GFDIFF", "GFFULL", "GFFULLDIFF", "F12V", "F12VBX", "ENERGY"};
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

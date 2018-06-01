// Copyright 2017

#include <array>
#include <string>
#include <vector>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif  // HAVE_CONFIG_H

#include "qc_mpi.h"

#ifndef QC_INPUT_H_
#define QC_INPUT_H_
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
  ORDER
};
}
namespace TASKS {
enum TaskVal {
  MP2 = 0,
  GF,
  GFDIFF,
  GFFULL,
  GFFULLDIFF
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
  std::string job_name;
};
#endif  // QC_INPUT_H_

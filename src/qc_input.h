#ifndef QC_INPUT_H_
#define QC_INPUT_H_

#include <array>
#include <string>
#include <vector>
#include <map>

#include "qc_mpi.h"
#include "tau_generators.h"
#include "samplers.h"

namespace KEYS {
  enum KEY_TYPE {
    STRING,
    INT,
    DOUBLE,
    BOOL,
    OTHER
  };
  enum KEYS {
#define FORMAT(X, Y) X,
#include "qc_input_keys.h"
#undef FORMAT
  };
  const std::map<KEYS, KEY_TYPE> KEY_TYPE_TABLE = {
#define FORMAT(X, Y) {X, Y},
#include "qc_input_keys.h"
#undef FORMAT
  };
  const std::vector<std::string> key_strings = {
#define FORMAT(X, Y) #X,
#include "qc_input_keys.h"
#undef FORMAT
  };
}
namespace JOBTYPE {
  enum JOBTYPE {
    ENERGY,
    DIMER,
    GF,
    GFDIFF,
    GFFULL,
    GFFULLDIFF,
  };
  const std::vector<std::string> jobtype_strings = {
    "ENERGY",
    "DIMER",
    "GF",
    "GFDIFF",
    "GFFULL",
    "GFFULLDIFF",
  };
}
namespace TASK {
  enum TASK {
    MP2 =         0b0000000001,
    MP3 =         0b0000000010,
    MP4 =         0b0000000100,
    MP2_F12_V =   0b0000001000,
    MP2_F12_VBX = 0b0000010000,
    GF2 =         0b0000100000,
    GF3 =         0b0001000000,
    GF2_F12_V =   0b0010000000,
    GF2_F12_VBX = 0b0100000000,
    DIRECT_MP2    = 0b1000000000,
    ANY_F12     = 0b0110011000,
    ANY_F12_VBX = 0b0100010000,
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
    "DIRECT_MP2",
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

#ifndef QC_MONTE_H_
#define QC_MONTE_H_

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <unordered_map>

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "qc_mpi.h"
#include "qc_input.h"
#include "molecule.h"
#include "basis/basis.h"
#include "weight_function.h"

#include "control_variate.h"
#include "electron_pair_list.h"
#include "electron_list.h"
#include "qc_ovps.h"
#include "qc_random.h"
#include "tau.h"

#include "MCGF/qc_mcgf.h"
#include "MCGF/qc_mcgf2.h"
#include "MCGF/qc_mcgf3.h"
#include "MCGF/gf_full_arrays.h"
#include "MCF12/gf2_f12.h"

class GFStats {
 public:
  std::vector<std::vector<double>> qeps;

  GFStats(bool, int, int, int, int, const std::string&, const std::string&);
  GFStats(bool, int, int, int, int, const std::string&, int);
  GFStats(const GFStats& gf) { throw std::runtime_error("Tried to copy GFStats"); }
  GFStats operator=(const GFStats& gf) { exit(0); }

  ~GFStats();

  void blockIt(const int&);
  void reduce();
  void print(const int& step, const double& time_span);

 private:
  std::vector<std::vector<double>> qepsEx1, qepsEx2, qepsAvg, qepsVar;
  std::vector<std::ofstream*> output_streams;
  bool isMaster;
  double tasks;
};

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class QC_monte {
 protected:
  typedef Basis<Container, Allocator> Basis_Type;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;
  typedef OVPS<Container, Allocator> OVPS_Type;

  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef Electron_List<Container, Allocator> Electron_List_Type;
  typedef Tau Tau_Type;

 public:
  QC_monte(MPI_info p0, IOPs p1, Molecule p2, Basis_Host p3);
  virtual ~QC_monte();
  virtual void monte_energy() = 0;

 protected:
  MPI_info mpi_info;
  IOPs iops;
  Molecule molec;
  Basis_Type basis;
  Electron_Pair_GTO_Weight electron_pair_weight;
  Electron_GTO_Weight electron_weight;

  std::unordered_map<int, Wavefunction_Type> wavefunctions;
  std::unordered_map<int, std::vector<int>> wavefunction_groups;

  Random random;
  OVPS_Type ovps;
  
  Electron_Pair_List_Type* electron_pair_list;
  Electron_List_Type* electron_list;
  Tau_Type* tau;

  int nDeriv;
  int numBand, offBand;
  int iocc1, iocc2, ivir1, ivir2;

  virtual void update_wavefunction();
  void move_walkers();
  static void print_mc_head(std::chrono::system_clock::time_point);
  static void print_mc_tail(double, std::chrono::system_clock::time_point);
};

template class QC_monte<std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class QC_monte<thrust::device_vector, thrust::device_allocator>;
#endif 

class GF : public QC_monte<std::vector, std::allocator> {
 public:
  void monte_energy() override;

 protected:
  GF(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : QC_monte(p1, p2, p3, p4) {}

  std::vector<GFStats> qeps;
  virtual void mc_local_energy(const int& step) = 0;
  virtual int full_print(int& step, int checkNum) = 0;
  std::string genFileName(int, int, int, int, int, int);

  void mcgf2_local_energy_core();
  void mcgf2_local_energy(std::vector<std::vector<double>>&);
  void mcgf2_local_energy_diff(std::vector<std::vector<double>>&);
  void mcgf2_local_energy_full(int);
  void mcgf2_local_energy_full_diff(int);

  void mcgf3_local_energy_core();
  void mcgf3_local_energy(std::vector<std::vector<double>>&);
  void mcgf3_local_energy_diff(std::vector<std::vector<double>>&);
  void mcgf3_local_energy_full(int);
  void mcgf3_local_energy_full_diff(int);

  void mc_gf_statistics(int, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&);
  void mc_gf_full_diffs(int band, std::vector<double> m);

  void mc_gf_copy(std::vector<double>&, std::vector<double>&);

  void mc_gf_full_print(int band, int steps, int checkNum, int order,
      std::vector<std::vector<double>>& d_ex1,
      std::vector<std::vector<std::vector<double>>>& d_cov);

  OVPS_ARRAY d_ovps;
};

class Diagonal_GF : public GF {
 public:
  Diagonal_GF(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4);
  void monte_energy() override;

 protected:
  std::vector<MCGF*> energy_functions;
  std::vector<GFStats> qeps;
  void mc_local_energy(const int& step);
  int full_print(int& step, int checkNum);
  std::string genFileName(int, int, int, int, int, int);
};

class GF2 : public GF {
 public:
  GF2(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : GF(p1, p2, p3, p4) {
    ovps.init(1, iops.iopns[KEYS::ELECTRON_PAIRS]);
    d_ovps.resize(iops, create_movec_parser(mpi_info, molec, MOVEC_TYPE::NWCHEM_BINARY, iops.sopns[KEYS::MOVECS], iops.bopns[KEYS::FREEZE_CORE]), {2});
    tau->resize(2);

    qeps.reserve(1);
    qeps.emplace_back(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, iops.sopns[KEYS::JOBNAME], 2);
  }
  ~GF2() {
  }
 protected:
  void mc_local_energy(const int& step);
  int full_print(int& step, int checkNum);
};

class GF3 : public GF {
 public:
  GF3(MPI_info p1, IOPs p2, Molecule p3, Basis_Host p4) : GF(p1, p2, p3, p4) {
    ovps.init(2, iops.iopns[KEYS::ELECTRON_PAIRS]);
    d_ovps.resize(iops, create_movec_parser(mpi_info, molec, MOVEC_TYPE::NWCHEM_BINARY, iops.sopns[KEYS::MOVECS], iops.bopns[KEYS::FREEZE_CORE]), {2, 3});
    tau->resize(2);

    qeps.reserve(2);
    qeps.emplace_back(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, iops.sopns[KEYS::JOBNAME], 2);
    qeps.emplace_back(mpi_info.sys_master, mpi_info.numtasks, numBand, offBand, nDeriv, iops.sopns[KEYS::JOBNAME], 3);
  }
  ~GF3() {}

 protected:
  void mc_local_energy(const int& step);
  int full_print(int& step, int checkNum);
};
#endif  // QC_MONTE_H_

// Copyright 2017

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "el_pair.h"
#include "weight_function.h"
#include "qc_basis.h"
#include "qc_geom.h"
#include "qc_input.h"
#include "qc_mpi.h"
#include "qc_ovps.h"
#include "qc_random.h"

#ifndef QC_MONTE_H_
#define QC_MONTE_H_
class GFStats {
 private:
  std::vector<std::vector<std::vector<double>>> qepsBlock, qepsEx1, qepsEx2, qepsAvg, qepsVar;
  std::vector<std::ofstream*> output_streams;
  bool isMaster;
  double tasks;

 public:
  std::vector<std::vector<double>> qeps;

  GFStats(bool, int, int, int, int, int, const std::string&, int);
  GFStats(const GFStats& gf) { exit(0); }
  GFStats operator=(const GFStats& gf) { exit(0); }

  ~GFStats();

  void blockIt(const int&);
  void reduce();
  void print(const int& step, const double& time_span);
};

class QC_monte {
 protected:
  std::vector<el_pair_typ> el_pair_list;

  int nDeriv, nBlock;
  int numBand, offBand;
  int iocc1, iocc2, ivir1, ivir2;
  const double rnd_min = 1.0e-3;
  int nsucc, nfail;
  double delx;

  MPI_info mpi_info;
  IOPs iops;
  Molec molec;
  Basis basis;
  GTO_Weight mc_basis;
  Random random;
  OVPs ovps;

  void update_wavefunction();
  void move_walkers();
  void scale_delx();
  void print_mc_head(std::chrono::high_resolution_clock::time_point);
  void print_mc_tail(double, std::chrono::high_resolution_clock::time_point);

  void mcgf2_local_energy_core();
  void mcgf2_local_energy(std::vector<double>&, int);
  void mcgf2_local_energy_diff(std::vector<double>&, int);
  void mcgf2_local_energy_full(int);
  void mcgf2_local_energy_full_diff(int);

  void mcgf3_local_energy_core();
  void mcgf3_local_energy(std::vector<double>&, int);
  void mcgf3_local_energy_diff(std::vector<double>&, int);
  void mcgf3_local_energy_full(int);
  void mcgf3_local_energy_full_diff(int);
  void mc_gf3_func(double*, int, int, int, int);

  void mc_gf_statistics(int,
                        std::vector<std::vector<double>>&,
                        std::vector<std::vector<double*>>&,
                        std::vector<std::vector<std::vector<double*>>>&,
                        std::vector<std::vector<std::vector<double*>>>&,
                        std::vector<std::vector<std::vector<double*>>>&);

  void mc_gf_copy(std::vector<double>&, std::vector<double>&, double*, double*);
  std::string genFileName(int, int, int, int, int, int);

  void mc_gf2_statistics(int, int);
  void mc_gf2_full_print(int, int, int);

  void mc_gf3_statistics(int, int);
  void mc_gf3_full_print(int, int, int);

 public:
  QC_monte(MPI_info p0, IOPs p1, Molec p2, Basis p3, GTO_Weight p4);
  ~QC_monte() {
    basis.gpu_free();
  }
  virtual void monte_energy() = 0;
};

class MP2 : public QC_monte {
 public:
  MP2(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.cpuMalloc_02(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                      iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                      iops.iopns[KEYS::NBLOCK], basis);
    ovps.gpuMalloc_02();

    lambda = 2.0 * (basis.nw_en[iocc2] - basis.nw_en[iocc2 - 1]);
    tau_values.resize(ivir2);
  }
  ~MP2() {
    ovps.cpuFree_02();
    ovps.gpuFree_02();
  }
  void monte_energy();

 protected:
  void mcmp2_energy(double&, std::vector<double>&);
  void new_tau();

  double lambda;
  double tau_wgt;
  std::vector<double> tau_values;
};

class MP3 : public QC_monte {
 public:
  MP3(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.cpuMalloc_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                      iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                      iops.iopns[KEYS::NBLOCK], basis);
    ovps.gpuMalloc_03();
  }
  ~MP3() {
    ovps.cpuFree_03();
    ovps.gpuFree_03();
  }
  void monte_energy();

 protected:
  void mcmp2_energy(double&, std::vector<double>&);
  void mcmp3_energy(double&, std::vector<double>&);
};

class GF2 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, int);

 public:
  GF2(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.cpuMalloc_02(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                      iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                      iops.iopns[KEYS::NBLOCK], basis);

    ovps.gpuMalloc_02();
#ifdef QUAD_TAU
    ovps.init_tau_02(basis);
#endif
  }
  ~GF2() {
    ovps.cpuFree_02();
    ovps.gpuFree_02();
  }
  void monte_energy();
};

class GF3 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, int);

 public:
  GF3(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.cpuMalloc_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                      iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                      iops.iopns[KEYS::NBLOCK], basis);
    ovps.gpuMalloc_03();
#ifdef QUAD_TAU
    ovps.init_tau_03(basis);
#endif
  }
  ~GF3() {
    ovps.cpuFree_03();
    ovps.gpuFree_03();
  }
  void monte_energy();
};
#endif  // QC_MONTE_H_

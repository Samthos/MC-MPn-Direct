// Copyright 2017
#ifndef QC_MONTE_H_
#define QC_MONTE_H_

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "el_pair.h"
#include "weight_function.h"
#include "basis/qc_basis.h"
#include "qc_geom.h"
#include "qc_input.h"
#include "qc_mpi.h"
#include "qc_ovps.h"
#include "qc_random.h"
#include "tau_integrals.h"

class GFStats {
 private:
  std::vector<std::vector<double>> qepsEx1, qepsEx2, qepsAvg, qepsVar;
  std::vector<std::ofstream*> output_streams;
  bool isMaster;
  double tasks;

 public:
  std::vector<std::vector<double>> qeps;

  GFStats(bool, int, int, int, int, const std::string&, int);
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

  MPI_info mpi_info;
  IOPs iops;
  Molec molec;
  Basis basis;
  GTO_Weight mc_basis;
  Random random;
  OVPs ovps;
  Stochastic_Tau tau;

  void update_wavefunction();
  void move_walkers();
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
                        std::vector<std::vector<double*>>&,
                        std::vector<std::vector<double*>>&);

  void mc_gf_copy(std::vector<double>&, std::vector<double>&, double*, double*);
  std::string genFileName(int, int, int, int, int);

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
    tau.resize(2, basis);
  }
  ~MP2() {
  }
  void monte_energy();

 protected:
  void mcmp2_energy(double&, std::vector<double>&);

  double lambda;
  double tau_wgt;
  std::vector<double> tau_values;
};

class MP3 : public QC_monte {
 public:
  MP3(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);
    ovps.alloc_03();
    tau.resize(2, basis);
  }
  ~MP3() {
    ovps.free_tau_03();
    ovps.free_03();
  }
  void monte_energy();

 protected:
  void mcmp2_energy(double&, std::vector<double>&);
  void mcmp3_energy(double&, std::vector<double>&);
};

class MP4 : public QC_monte {
 public:
  MP4(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);
    ovps.alloc_03();
  }
  ~MP4() {
    ovps.free_tau_03();
    ovps.free_03();
  }
  void monte_energy();

 protected:
  void mcmp2_energy(double&, std::vector<double>&);
  void mcmp3_energy(double&, std::vector<double>&);
  void mcmp4_energy(double&, std::vector<double>&);
};

class GPU_GF2 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, int);

 public:
  GPU_GF2(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_02(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);

    ovps.alloc_02();
  }
  ~GPU_GF2() {
    ovps.free_tau_02();
    ovps.free_02();
  }
  void monte_energy();
};

class GF2 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, int);

 public:
  GF2(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_02(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);

    ovps.alloc_02();
    tau.resize(2, basis);
  }
  ~GF2() {
    ovps.free_tau_02();
    ovps.free_02();
  }
  void monte_energy();
};

class GPU_GF3 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, int);

 public:
  GPU_GF3(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);
    ovps.alloc_03();
  }
  ~GPU_GF3() {
    ovps.free_tau_03();
    ovps.free_03();
  }
  void monte_energy();
};

class GF3 : public QC_monte {
 protected:
  void mc_local_energy(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, int);

 public:
  GF3(MPI_info p1, IOPs p2, Molec p3, Basis p4, GTO_Weight p5) : QC_monte(p1, p2, p3, p4, p5) {
    ovps.init_03(iops.iopns[KEYS::MC_NPAIR], iops.iopns[KEYS::NUM_BAND],
                 iops.iopns[KEYS::OFF_BAND], iops.iopns[KEYS::DIFFS],
                 basis);
    ovps.alloc_03();
    tau.resize(2, basis);
  }
  ~GF3() {
    ovps.free_tau_03();
    ovps.free_03();
  }
  void monte_energy();
};
#endif  // QC_MONTE_H_
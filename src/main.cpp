// Copyright 2017

#include "weight_function.h"
#include "mpi.h"
#include "qc_basis.h"
#include "qc_geom.h"
#include "qc_input.h"
#include "qc_monte.h"
#include "qc_mpi.h"
#include "qc_ovps.h"

#define VERSION "fix"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_info mpi_info;

  if (argc != 2) {
    if (mpi_info.sys_master) {
      printf("Usage: mcmpN.x <input>\n");
    }
    exit(EXIT_FAILURE);
  } else {
    if (mpi_info.sys_master) {
      printf("MC-GFn program developed by the Hirata lab\n");
      printf("Code compiled from Git-Commit %s\n\n", VERSION);
    }
  }
  mpi_info.print();

  IOPs iops;
  iops.read(mpi_info, argv[1]);
  iops.print(mpi_info, argv[1]);

  Molec molec;
  molec.read(mpi_info, iops.sopns[KEYS::GEOM]);

  Basis basis;
  basis.read(iops, mpi_info, molec);
  basis.nw_vectors_read(mpi_info, molec, iops);

  GTO_Weight mc_basis;
  mc_basis.read(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]);

  if (iops.iopns[KEYS::TASK] == TASKS::MP) {
    if (iops.iopns[KEYS::ORDER] == 2) {
      MP2 qc_monte(mpi_info, iops, molec, basis, mc_basis);
      qc_monte.monte_energy();
    } else if (iops.iopns[KEYS::ORDER] == 3) {
      MP3 qc_monte(mpi_info, iops, molec, basis, mc_basis);
      qc_monte.monte_energy();
    }
  } else {
    if (iops.iopns[KEYS::ORDER] == 2) {
      GF2 qc_monte(mpi_info, iops, molec, basis, mc_basis);
      qc_monte.monte_energy();
    } else if (iops.iopns[KEYS::ORDER] == 3) {
      GF3 qc_monte(mpi_info, iops, molec, basis, mc_basis);
      qc_monte.monte_energy();
    }
  }

  MPI_Finalize();
}

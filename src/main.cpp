// Copyright 2017

#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "weight_function.h"
#include "basis/qc_basis.h"
#include "qc_monte.h"

int main(int argc, char* argv[]) {
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
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

  Basis basis(iops, mpi_info, molec);

  GTO_Weight mc_basis;
  mc_basis.read(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]);

  auto el_pair_list = create_sampler(iops, molec, mc_basis);

  if (iops.iopns[KEYS::TASK] == TASKS::MP) {
    if (iops.iopns[KEYS::ORDER] == 2) {
      MP2 qc_monte(mpi_info, iops, molec, basis, mc_basis, el_pair_list);
      qc_monte.monte_energy();
    } else if (iops.iopns[KEYS::ORDER] == 3) {
      MP3 qc_monte(mpi_info, iops, molec, basis, mc_basis, el_pair_list);
      qc_monte.monte_energy();
    } else if (iops.iopns[KEYS::ORDER] == 4) {
      MP4 qc_monte(mpi_info, iops, molec, basis, mc_basis, el_pair_list);
      qc_monte.monte_energy();
    }
  } else {
    if (iops.iopns[KEYS::ORDER] == 2) {
      GF2 qc_monte(mpi_info, iops, molec, basis, mc_basis, el_pair_list);
      qc_monte.monte_energy();
    } else if (iops.iopns[KEYS::ORDER] == 3) {
      GF3 qc_monte(mpi_info, iops, molec, basis, mc_basis, el_pair_list);
      qc_monte.monte_energy();
    }
  }

  destroy_sampler(el_pair_list, static_cast<SAMPLERS::SAMPLERS>(iops.iopns[KEYS::SAMPLER]));

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
}

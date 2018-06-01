// Copyright 2017

#include <cstdio>

#include "mpi.h"

#include "qc_mpi.h"

MPI_info::MPI_info() {
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  if (0 == taskid) {
    sys_master = true;
  } else {
    sys_master = false;
  }
}

void MPI_info::mpi_set_info() {
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  if (0 == taskid) {
    sys_master = true;
  } else {
    sys_master = false;
  }
}

void MPI_info::print() {
  if (sys_master) {
    printf("MPI IS RUNNING WITH %i CPUS\n", numtasks);
  }
}

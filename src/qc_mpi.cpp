// Copyright 2017

#include <cstdio>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "qc_mpi.h"

MPI_info::MPI_info() {
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#else
  numtasks = 1;
  taskid = 0;
#endif

  if (0 == taskid) {
    sys_master = true;
  } else {
    sys_master = false;
  }
}

void MPI_info::mpi_set_info() {
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#else
  numtasks = 1;
  taskid = 0;
#endif

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

// Copyright 2017

#include <cstdio>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "qc_mpi.h"

MPI_info::MPI_info() {
  /*
   * MPI_info constructor
   */
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#else
  numtasks = 1;
  taskid = 0;
#endif

  sys_master = 0 == taskid;
}

void MPI_info::print() {
  /*
   * Print number of mpi tasks if sys_master
   */
  if (sys_master) {
    printf("MPI IS RUNNING WITH %i CPUS\n", numtasks);
  }
}

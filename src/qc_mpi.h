// Copyright 2017
#ifndef QC_MPI_H_
#define QC_MPI_H_
class MPI_info {
 public:
  MPI_info();
  void mpi_set_info();
  void print();
  int numtasks;  // formerly sys_nproc
  int taskid;  // formerly sys_me

  double dnumtasks;  // double version of nubmer of tasks
  bool sys_master;
};
#endif  // QC_MPI_H_

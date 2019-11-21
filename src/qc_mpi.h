// Copyright 2019
#ifndef QC_MPI_H_
#define QC_MPI_H_
class MPI_info {
 public:
  MPI_info();
  void print();

  int numtasks;
  int taskid;
  bool sys_master;
};
#endif  // QC_MPI_H_

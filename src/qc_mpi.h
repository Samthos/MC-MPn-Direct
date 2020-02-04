// Copyright 2019
#ifndef QC_MPI_H_
#define QC_MPI_H_
#include <vector>

class MPI_info {
 public:
  MPI_info();
  void print();

  static void barrier();
  static void broadcast_int(int*, size_t);
  static void broadcast_char(char*, size_t);
  static void broadcast_double(double*, size_t);
  static void broadcast_vector_double(std::vector<double>&);

  static void reduce_long_long_uint(long long unsigned int* source, long long unsigned int* dest, size_t size);
  static void reduce_double(double* source, double* dest, size_t size);

  static void comm_rank(int*);
  static void comm_size(int*);

  int numtasks;
  int taskid;
  bool sys_master;
};
#endif  // QC_MPI_H_

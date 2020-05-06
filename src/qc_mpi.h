// Copyright 2019
#ifndef QC_MPI_H_
#define QC_MPI_H_

#include <string>
#include <vector>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

class MPI_info {
 public:
  MPI_info();
  void print();

  static void barrier();
  static void broadcast_int(int*, size_t);
  static void broadcast_char(char*, size_t);
  static void broadcast_double(double*, size_t);
  static void broadcast_vector_double(std::vector<double>&);
  static void broadcast_string(std::string&);

  template <class T>
  static void scatter_vector(std::vector<T>& v, T* w) {
#ifdef HAVE_MPI
    MPI_Scatter(
        v.data(), sizeof(T), MPI_CHAR,
        w, sizeof(T), MPI_CHAR, 
        0, MPI_COMM_WORLD);
#else
    *w = v[0];
#endif
  }

  template <class T>
  static void gather_vector(T* w, std::vector<T>& v) {
#ifdef HAVE_MPI
    MPI_Gather(
        w, sizeof(T), MPI_CHAR, 
        v.data(), sizeof(T), MPI_CHAR,
        0, MPI_COMM_WORLD);
#else
    v[0] = *w;
#endif
  }

  static void reduce_long_long_uint(long long unsigned int* source, long long unsigned int* dest, size_t size);
  static void reduce_double(double* source, double* dest, size_t size);

  static void comm_rank(int*);
  static void comm_size(int*);

  int numtasks;
  int taskid;
  bool sys_master;
};
#endif  // QC_MPI_H_

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
  comm_size(&numtasks);
  comm_rank(&taskid);
  sys_master = (0 == taskid);
}

void MPI_info::print() {
  /*
   * Print number of mpi tasks if sys_master
   */
  if (sys_master) {
    printf("MPI IS RUNNING WITH %i CPUS\n", numtasks);
  }
}

void MPI_info::barrier() {
#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void MPI_info::broadcast_int(int* v, size_t size) {
#ifdef HAVE_MPI
  MPI_Bcast(v, size, MPI_INT, 0, MPI_COMM_WORLD);
#endif
}

void MPI_info::broadcast_char(char* v, size_t size) {
#ifdef HAVE_MPI
  MPI_Bcast(v, size, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
}

void MPI_info::broadcast_double(double* v, size_t size) {
#ifdef HAVE_MPI
  MPI_Bcast(v, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

void MPI_info::broadcast_vector_double(std::vector<double>& v) {
#ifdef HAVE_MPI
  MPI_Bcast(v.data(), v.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

void MPI_info::broadcast_string(std::string& v) {
#ifdef HAVE_MPI
  int size = v.size();
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  v.resize(size);
  char *w = new char[size];
  std::copy(v.begin(), v.end(), w);

  MPI_Bcast(w, size, MPI_CHAR, 0, MPI_COMM_WORLD);
  std::copy(w, w + size, v.begin());
  delete w;
#endif
}

void MPI_info::reduce_long_long_uint(long long unsigned int* source, long long unsigned int* dest, size_t size) {
#ifdef HAVE_MPI
  MPI_Reduce(source, dest, size, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  std::copy(source, source + size, dest);
#endif
}

void MPI_info::reduce_double(double* source, double* dest, size_t size) {
#ifdef HAVE_MPI
  MPI_Reduce(source, dest, size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  std::copy(source, source + size, dest);
#endif
}

void MPI_info::comm_rank(int* v) {
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, v);
#else
  *v= 0;
#endif
}

void MPI_info::comm_size(int* v) {
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, v);
#else
  *v = 1;
#endif
}

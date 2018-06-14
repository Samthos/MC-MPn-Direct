//
// Created by aedoran on 5/31/18.
//

#ifndef MC_MP2_DIRECT_TIMER_H
#define MC_MP2_DIRECT_TIMER_H

#include <chrono>
#ifdef HAVE_MPI
#include "mpi.h"
#endif

class Timer {
 public:
  Timer() {
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &master);
#else
    master = 0;
#endif
  }
  void Start() {
    start = std::chrono::high_resolution_clock::now();
  }
  void Stop() {
    stop = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
  }
  friend std::ostream& operator<< (std::ostream& os, Timer& timer) {
    timer.Stop();
    if (0 == timer.master) {
      os << std::fixed << std::showpos << std::setprecision(7) << timer.span.count();
    }
    return os;
  }
 private:
  int master;
  std::chrono::high_resolution_clock::time_point start, stop;
  std::chrono::duration<double> span;
};

#endif //MC_MP2_DIRECT_TIMER_H

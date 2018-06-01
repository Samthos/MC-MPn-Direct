// Copyright 2017

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <iostream>
#include <sstream>
#include <string>

#ifdef USE_MPI
#include "mpi.h"
#endif

#include "qc_random.h"

void Random::seed(int param) {
  int taskid;
  uniform_distribution = std::uniform_real_distribution<double>(0.00, 1.00);

  debug = param;
  if (0 == debug) {
    std::string str;
    std::stringstream sstr;
    std::random_device r1;
    sstr << r1();
    sstr >> str;

    std::seed_seq seed1(str.begin(), str.end());
    g1.seed(seed1);
  } else if (1 == debug) {
    std::string str;
    std::stringstream sstr;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#else
    taskid = 0;
#endif
    long long seeds[] = {1793817729, 3227188512, 2530944000, 2295088101,
                         1099163413, 2366715906, 1817017634, 3192454568,
                         2199752324, 1911074146, 2867042420, 3591224665,
                         8321197510, 1781877856, 1785033536, 3983696536};
    sstr << seeds[taskid];
    sstr >> str;
    std::seed_seq seed1(str.begin(), str.end());
    g1.seed(seed1);
  }
}

double Random::get_rand() {
  return uniform_distribution(g1);
}
double Random::uniform(double low, double high) {
  std::uniform_real_distribution<double> dist(low, high);
  return dist(g1);
}
double Random::normal(double mu, double sigma) {
  std::normal_distribution<double> dist(mu, sigma);
  return dist(g1);
}

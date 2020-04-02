// Copyright 2017

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "qc_mpi.h"

#include "qc_random.h"

Random::Random(int param) {
  int taskid;
  uniform_distribution = std::uniform_real_distribution<double>(0.00, 1.00);
  normal_distribution = std::normal_distribution<double>(0.0, 1.0);

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
    MPI_info::comm_rank(&taskid);
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

double Random::uniform() {
  return uniform_distribution(g1);
}
double Random::uniform(double low, double high) {
  return (high-low) * uniform() + low;
}
double Random::normal(double mu, double sigma) {
  return sigma * normal_distribution(g1) + mu;
}

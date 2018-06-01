// Copyright 2017
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "mpi.h"

#include "qc_random.h"

Random::Random() {
}
Random::~Random() {
}

void Random::seed(int param) {
  int taskid;
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

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
    uint64_t seeds[] = {1793817729, 3227188512, 2530944000, 2295088101,
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
  static std::uniform_real_distribution<double> dist(0.00, 1.00);
  return dist(g1);
}
double Random::get_rand_exp(double param) {
  static std::exponential_distribution<double> dist(param);
  return dist(g1);
}
void Random::get_rand(double* param) {
  static std::uniform_real_distribution<double> dist(0.00, 1.00);
  *(param) = dist(g1);
}
void Random::get_rand3(double* param) {
  static std::uniform_real_distribution<double> dist(0.00, 1.00);
  *(param) = dist(g1);
  *(param + 1) = dist(g1);
  *(param + 2) = dist(g1);
}

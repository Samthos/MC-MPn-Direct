// Copyright 2017

#include <random>

#ifndef QC_RANDOM_H_
#define QC_RANDOM_H_
class Random {
 private:
  std::mt19937 g1;
  int debug;

 public:
  Random();
  ~Random();

  void seed(int);

  double get_rand();
  double get_rand_exp(double);
  void get_rand(double*);
  void get_rand3(double*);
};
#endif  // QC_RANDOM_H_

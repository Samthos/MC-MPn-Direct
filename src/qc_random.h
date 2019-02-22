// Copyright 2017
#pragma once
#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <array>
#include <random>

#ifndef QC_RANDOM_H_
#define QC_RANDOM_H_
class Random {
 public:
  explicit Random(int);
  double get_rand();
  double uniform(double, double);
  double normal(double, double);

 private:
  std::mt19937 g1;
  std::uniform_real_distribution<double> uniform_distribution;
  int debug;
};
#endif  // QC_RANDOM_H_

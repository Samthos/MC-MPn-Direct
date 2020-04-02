//
// Created by aedoran on 5/25/18.
//

#ifndef MC_MP2_DIRECT_CONTROL_VARIATE_H
#define MC_MP2_DIRECT_CONTROL_VARIATE_H

#define ARMA_DONT_PRINT_ERRORS

#include <functional>
#include <algorithm>
#include <armadillo>
#include <iomanip>

#include "qc_mpi.h"

class Accumulator {
 public:
  Accumulator();
  virtual void add(double x, const std::vector<double>& c) = 0;
  virtual void update() = 0;
  virtual void to_json(std::string fname) = 0;
  virtual std::ostream& write(std::ostream& os) = 0;
  virtual ~Accumulator() = default;

  friend std::ostream& operator << (std::ostream& os, Accumulator& accumulator) {
    return accumulator.write(os);
  }
 protected:
  int master;
  int comm_size;
  unsigned long long int n_samples;
  unsigned long long int total_samples;
};

Accumulator* create_accumulator(const bool& requires_blocking, const std::vector<double>& Exact_CV);

class Simple_Accumulator : public Accumulator {
 public:
  void add(double x, const std::vector<double>& c);
  void update();
  void to_json(std::string fname);
  std::ostream& write(std::ostream& os);
 private:
  arma::vec2 s_x;
  arma::vec2 e_x;
  double var, std, error;
};

class ControlVariate : public Accumulator {
 public:
  ControlVariate(size_t nControlVariates, const std::vector<double>& ExactCV);
  ControlVariate();
  ~ControlVariate() = default;

  // getters
  unsigned long long int getNumberOfSteps();
  size_t size();
  double getSx(int i);
  double getSCV(int i);
  double getSCVCov(int row, int col);
  double getSXCVCov(int row);

  double getEx(int i=0);
  double getVar();
  double getStd();
  double getError();

  double getCV(int i);
  double getCVCov(size_t row, size_t col);
  double getXCVCov(size_t row);
  double getCVEstimate();

  double getCVVariances();
  double getCVStd();
  double getCVError(); 
  double getExactCV(int i);

  void add(double x, const std::vector<double>& c) override;
  void update() override;
  void to_json(std::string fname) override; 
  std::ostream& write(std::ostream& os) override;

 private:
  // accumulators
  arma::vec2 s_x;
  arma::vec s_c1;
  arma::vec s_xc;

  arma::mat s_c2;

  // averages
  arma::vec2 e_x;
  arma::vec e_c1;
  arma::vec e_xc;
  arma::vec cov_xc;

  arma::mat e_c2;
  arma::mat cov_c;

  // statistics
  double var, std, error;
  double e_cv, var_cv, std_cv, error_cv;

  // exact value of control variates
  arma::vec exact_cv;
  arma::Col<double> alpha;
};

class BlockingAccumulator : public Accumulator {
 public:
  BlockingAccumulator(size_t nControlVariates, const std::vector<double>& ExactCV);
  BlockingAccumulator();
  ~BlockingAccumulator() = default;

  void resize(int new_size);
  void add(double x, const std::vector<double>& c) override;
  void update() override;
  void to_json(std::string fname) override;
  std::ostream& write(std::ostream& os) override;
 private:
  arma::vec s_x1;
  arma::vec s_x2;
  arma::vec s_block;

  // averages
  arma::vec e_x1;
  arma::vec e_x2;

  // statistics
  arma::vec var, std, error;
};
#endif //MC_MP2_DIRECT_CONTROL_VARIATE_H

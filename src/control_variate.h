//
// Created by aedoran on 5/25/18.
//

#ifndef MC_MP2_DIRECT_CONTROL_VARIATE_H
#define MC_MP2_DIRECT_CONTROL_VARIATE_H

#include <functional>
#include <algorithm>
#include <armadillo>

#ifdef USE_MPI
#include "mpi.h"
#endif

class ControlVariate {
 public:
  ControlVariate(size_t nControlVariates, const std::vector<double>& ExactCV) {
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &master);
#else
    master = 0;
#endif

    nSamples = 0;
    exact_cv = ExactCV;

    s_x.zeros();

    s_c1.zeros(nControlVariates);
    e_c1.zeros(nControlVariates);

    s_xc.zeros(nControlVariates);
    e_xc.zeros(nControlVariates);
    cov_xc.zeros(nControlVariates);

    e_c2.zeros(nControlVariates, nControlVariates);
    s_c2.zeros(nControlVariates, nControlVariates);
    cov_c.zeros(nControlVariates, nControlVariates);
  }
  ControlVariate() : ControlVariate(0, {}) {}

  // getters
  unsigned long long int getNumberOfSteps() {
    return nSamples;
  }
  size_t size() {
    return s_c1.size();
  }
  double getSx(int i) {
    return s_x[i];
  }
  double getSCV(int i) {
    return s_c1[i];
  }
  double getSCVCov(int row, int col) {
    return s_c2(row, col);
  }
  double getSXCVCov(int row) {
    return s_xc[row];
  }

  double getEx(int i=0) {
    update();
    return e_x[i];
  }
  double getVar() {
    update();
    return var;
  }
  double getStd() { update();
    return std;
  }
  double getError() {
    update();
    return error;
  }

  double getCV(int i) {
    update();
    return e_c1[i];
  }
  double getCVCov(size_t row, size_t col) {
    update();
    return cov_c(row, col);
  }
  double getXCVCov(size_t row) {
    update();
    return cov_xc[row];
  }
  double getCVEstimate() {
    update();
    return e_cv;
  }

  double getCVVariances() {
    update();
    return var_cv;
  }
  double getCVStd() {
    update();
    return std_cv;
  }
  double getCVError() {
    update();
    return error_cv;
  }

  double getExactCV(int i) {
    update();
    return exact_cv[i];
  }

  void add(double x, const std::vector<double>& c) {
    arma::vec c_(c);

    s_x[0] += x;
    s_x[1] += x*x;

    s_c1 += c_;
    s_xc += x * c_;

    s_c2 += c_ * c_.t();
    nSamples++;
  }
  double update() {
    // calculate averages
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&nSamples, &TotalSamples, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_x.memptr(), e_x.memptr(), s_x.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_c1.memptr(), e_c1.memptr(), s_c1.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_c2.memptr(), e_c2.memptr(), s_c2.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_xc.memptr(), e_xc.memptr(), s_xc.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    e_x  = e_x  / static_cast<double>(TotalSamples);
    e_c1 = e_c1 / static_cast<double>(TotalSamples);
    e_c2 = e_c2 / static_cast<double>(TotalSamples);
    e_xc = e_xc / static_cast<double>(TotalSamples);
#else
    TotalSamples = nSamples;
    e_x  = s_x  / static_cast<double>(nSamples);
    e_c1 = s_c1 / static_cast<double>(nSamples);
    e_c2 = s_c2 / static_cast<double>(nSamples);
    e_xc = s_xc / static_cast<double>(nSamples);
#endif

    if (0 == master) {
      // calculate variance and derived
      var = e_x[1] - e_x[0] * e_x[0];
      std = sqrt(var);
      error = sqrt(var / static_cast<double>(TotalSamples));

      // calculate covariance of control variates
      cov_xc = e_xc - e_c1 * e_x[0];
      cov_c = e_c2 - e_c1 * e_c1.t();

      // calcalate scaling coeficent for contrl variates
      arma::Col<double> alpha = arma::solve(cov_c, cov_xc);

      // calcaulte control varaiate averate
      e_cv = e_x[0] - arma::dot(alpha, e_c1 - exact_cv);

      // calcaulte statistics for control variate guess
      var_cv = var - arma::dot(cov_xc, alpha);
      std_cv = sqrt(var_cv);
      error_cv = sqrt(var_cv / static_cast<double>(TotalSamples));
    }
  }

  friend std::ostream& operator<< (std::ostream& os, ControlVariate& cv) {
    cv.update();
    if (0 == cv.master) {
      os << cv.nSamples << "\t";
      os << std::setprecision(7);
      os << cv.e_x[0] << "\t";
      os << cv.error << "\t";
      os << cv.e_cv << "\t";
      os << cv.error_cv;
    }
    return os;
  }
 private:
  int master;
  unsigned long long int nSamples;
  unsigned long long int TotalSamples;
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
};


#endif //MC_MP2_DIRECT_CONTROL_VARIATE_H
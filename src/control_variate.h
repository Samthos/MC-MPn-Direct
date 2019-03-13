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

#ifdef HAVE_MPI
#include "mpi.h"
#endif

class Accumulator {
 public:
  virtual size_t size() = 0;
  virtual void add(double x, const std::vector<double>& c) = 0;
  virtual void update() = 0;
  virtual void to_json(std::string fname) = 0;
  virtual std::ostream& write(std::ostream& os) = 0;

  friend std::ostream& operator << (std::ostream& os, Accumulator& accumulator) {
    return accumulator.write(os);
  }
 private:
};

class ControlVariate : public Accumulator {
 public:
  ControlVariate(size_t nControlVariates, const std::vector<double>& ExactCV) {
#ifdef HAVE_MPI
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

  void add(double x, const std::vector<double>& c) override {
    arma::vec c_(c);

    s_x[0] += x;
    s_x[1] += x*x;

    s_c1 += c_;
    s_xc += x * c_;

    s_c2 += c_ * c_.t();
    nSamples++;
  }
  void update() override {
    // calculate averages
#ifdef HAVE_MPI
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
      alpha = arma::solve(cov_c, cov_xc);

      // calcaulte control varaiate averate
      e_cv = e_x[0] - arma::dot(alpha, e_c1 - exact_cv);

      // calcaulte statistics for control variate guess
      var_cv = var - arma::dot(cov_xc, alpha);
      std_cv = sqrt(var_cv);
      error_cv = sqrt(var_cv / static_cast<double>(TotalSamples));
    }
  }
  void to_json(std::string fname) override {
    // open stream
    std::ofstream os(fname + ".json");

    // call update function
    update();

    if (0 == master) {
      // open json
      os << "{\n";

      // print number of steps
      os << "\t\"Steps\" : " << TotalSamples << ",\n";

      // print E[x]
      os << "\t\"EX\" : " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_x[0] << ",\n";

      // print Var[x]
      os << "\t\"EX2\" : " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_x[1] << ",\n";

      // print vector of E[cv]
      os << "\t\"EC\" : [" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_c1[0];
      for (auto i = 1; i < e_c1.size(); i++) {
        os << ",\n\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_c1[i];
      }
      os << "\n\t],\n";

      // print E[x * cv]
      os << "\t\"EXC\" : [" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_xc[0];
      for (auto i = 1; i < e_xc.size(); i++) {
        os << ",\n\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_xc[i];
      }
      os << "\n\t],\n";

      // print Cov[x * cv]
      os << "\t\"COVXC\" : [" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << cov_xc[0];
      for (auto i = 1; i < cov_xc.size(); i++) {
        os << ",\n\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << cov_xc[i];
      }
      os << "\n\t],\n";

      // print alpha
      os << "\t\"alpha\" : [" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << alpha[0];
      for (auto i = 1; i < alpha.size(); i++) {
        os << ",\n\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << alpha[i];
      }
      os << "\n\t],\n";

      // print E[cv.T * cv]
      os << "\t\"ECC\" : [";
      for (auto row = 0; row < e_c2.n_rows; row++) {
        if (row != 0) {
          os << "],";
        }

        os << "\n\t\t[" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_c2(row, 0);
        for (auto col = 1; col < e_c2.n_cols; col++) {
          os << ", " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_c2(row, col);
        }
      }
      os << "]],\n";

      // print E[cv.T * cv]
      os << "\t\"COVCC\" : [";
      for (auto row = 0; row < cov_c.n_rows; row++) {
        if (row != 0) {
          os << "],";
        }

        os << "\n\t\t[" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << cov_c(row, 0);
        for (auto col = 1; col < cov_c.n_cols; col++) {
          os << ", " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << cov_c(row, col);
        }
      }
      os << "]]\n";

      os << "}";
    }

    // close stream
    os.close();
  }
  std::ostream& write(std::ostream& os) override {
    update();
    if (0 == master) {
#ifndef FULL_PRINTING
      os << nSamples << "\t";
      os << std::setprecision(7);
      os << e_x[0] << "\t";
      os << error << "\t";
      os << e_cv << "\t";
      os << error_cv;
#else
      os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_x[0] << ",";
      os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << error << ",";
#endif
    }
    return os;
  }

  friend std::ostream& operator<< (std::ostream& os, ControlVariate& cv) {
    return cv.write(os);
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
  arma::Col<double> alpha;
};

class BlockingAccumulator : public Accumulator {
 public:
  BlockingAccumulator(size_t nControlVariates, const std::vector<double>& ExactCV) {
    master = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &master);
#endif
    nSamples = 0;
    TotalSamples = 0;

    s_x1 = 0;
    s_x2.resize(1);
    s_x2.fill(0);

    e_x1 = 0;
  }
  BlockingAccumulator() : BlockingAccumulator(0, {}) {}

  void add(double x, const std::vector<double>& c) override {
    uint32_t block = 0;
    uint32_t blockPower2 = 1;

    nSamples++;

    s_x1 += x;
    s_x2[0] += x*x;

    if (block < s_block.size()-1) {
      s_block.resize(block+2);
      s_x2.resize(block+2);
    }
    s_block[block+1] += x;

    block++;
    blockPower2 *= 2;
    while ((nSamples & (blockPower2-1)) == 0 && block < s_block.size()) {
      s_block[block] /= 2;
      s_x2[block] += s_block[block] * s_block[block];

      if (block < s_block.size()-1) {
        s_block.resize(block+2);
        s_x2.resize(block+2);
      }
      s_block[block+1] += s_block[block];

      s_block[block] = 0;

      block++;
      blockPower2 *= 2;
    }
  }
  void update() override {
    // calculate averages
    if (s_x2.size() != e_x2.size()) {
      e_x2.resize(s_x2.size());
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&nSamples, &TotalSamples, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&s_x1, &e_x1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(s_x2.memptr(), e_x2.memptr(), s_x2.n_elem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    e_x1  = e_x1 / static_cast<double>(TotalSamples);
    e_x2  = e_x2 / static_cast<double>(TotalSamples);
#else
    TotalSamples = nSamples;
    e_x1  = s_x1 / static_cast<double>(TotalSamples);
    e_x2  = s_x2 / static_cast<double>(TotalSamples);
#endif

    if (0 == master) {
      // calculate variance and derived
      var = e_x2 - e_x1 * e_x1;
      std = sqrt(var);
      error = sqrt(var / static_cast<double>(TotalSamples));
    }
  }
  void to_json(std::string fname) override {
    // open stream
    std::ofstream os(fname + ".json");

    // call update function
    update();

    if (0 == master) {
      // open json
      os << "{\n";

      // print number of steps
      os << "\t\"Steps\" : " << TotalSamples << ",\n";

      // print E[x]
      os << "\t\"EX\" : " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_x1 << ",\n";

      // print Var[x]
      os << "\t\"EX2\" : " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << e_x2 << ",\n";

      os << "}";
    }

    // close stream
    os.close();
  }
  std::ostream& write(std::ostream& os) override {
    update();
    if (0 == master) {
      os << nSamples << "\t";
      os << std::setprecision(7);
      os << e_x1 << "\t";
      os << error;
    }
    return os;
  }
 private:
  int master;
  unsigned long long int nSamples;
  unsigned long long int TotalSamples;

  double s_x1;
  arma::vec s_x2;
  arma::vec s_block;

  // averages
  double e_x1;
  arma::vec e_x2;

  // statistics
  arma::vec var, std, error;
};
#endif //MC_MP2_DIRECT_CONTROL_VARIATE_H

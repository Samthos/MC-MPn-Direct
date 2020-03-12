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
  virtual void add(double x, const std::vector<double>& c) = 0;
  virtual void update() = 0;
  virtual void to_json(std::string fname) = 0;
  virtual std::ostream& write(std::ostream& os) = 0;
  virtual ~Accumulator() = default;

  friend std::ostream& operator << (std::ostream& os, Accumulator& accumulator) {
    return accumulator.write(os);
  }
 private:
};

Accumulator* create_accumulator(const bool& requires_blocking, const std::vector<double>& Exact_CV);

class ControlVariate : public Accumulator {
 public:
  ControlVariate(size_t nControlVariates, const std::vector<double>& ExactCV) {
    MPI_info::comm_rank(&master);

    n_samples = 0;
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
  ~ControlVariate() = default;

  // getters
  unsigned long long int getNumberOfSteps() {
    return n_samples;
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
    n_samples++;
  }
  void update() override {
    // calculate averages
    MPI_info::barrier();
    MPI_info::reduce_long_long_uint(&n_samples, &total_samples, 1);
    MPI_info::reduce_double(s_x.memptr(), e_x.memptr(), s_x.n_elem);
    MPI_info::reduce_double(s_c1.memptr(), e_c1.memptr(), s_c1.n_elem);
    MPI_info::reduce_double(s_c2.memptr(), e_c2.memptr(), s_c2.n_elem);
    MPI_info::reduce_double(s_xc.memptr(), e_xc.memptr(), s_xc.n_elem);
    e_x  = e_x  / static_cast<double>(total_samples);
    e_c1 = e_c1 / static_cast<double>(total_samples);
    e_c2 = e_c2 / static_cast<double>(total_samples);
    e_xc = e_xc / static_cast<double>(total_samples);

    if (0 == master) {
      // calculate variance and derived
      var = e_x[1] - e_x[0] * e_x[0];
      std = sqrt(var);
      error = sqrt(var / static_cast<double>(total_samples));

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
      error_cv = sqrt(var_cv / static_cast<double>(total_samples));
    }
  }
  void to_json(std::string fname) override {
    // call update function
    update();

    if (0 == master) {
      // open stream
      std::ofstream os(fname + ".json");

      // open json
      os << "{\n";

      // print number of steps
      os << "\t\"Steps\" : " << total_samples << ",\n";

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
  }
  std::ostream& write(std::ostream& os) override {
    update();
    if (0 == master) {
#ifndef FULL_PRINTING
      os << n_samples << "\t";
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

 private:
  int master;
  unsigned long long int n_samples;
  unsigned long long int total_samples;
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
    MPI_info::comm_rank(&master);
    MPI_info::comm_size(&comm_size);

    n_samples = 0;
    total_samples = 0;

    resize(1);
  }
  BlockingAccumulator() : BlockingAccumulator(0, {}) {}
  ~BlockingAccumulator() = default;

  void resize(int new_size) {
    int old_size = s_x1.n_elem;
    s_x1.resize(new_size);
    e_x1.resize(new_size);
    s_x2.resize(new_size);
    s_block.resize(new_size);
    e_x2.resize(new_size);
    var.resize(new_size);
    std.resize(new_size);
    error.resize(new_size);
    for (; old_size < new_size; old_size++) {
      s_x1[old_size] = 0;
      e_x1[old_size] = 0;
      s_x2[old_size] = 0;
      s_block[old_size] = 0;
      e_x2[old_size] = 0;
      var[old_size] = 0;
      std[old_size] = 0;
      error[old_size] = 0;
    }
  }
  void add(double x, const std::vector<double>& c) override {
    uint32_t block = 0;
    uint32_t blockPower2 = 1;

    n_samples++;

    s_x1[0] += x;
    s_x2[0] += x*x;

    if (s_block.n_elem-1 <= block) {
      resize(block+2);
    }
    s_block[block+1] += x;

    block++;
    blockPower2 *= 2;
    while ((n_samples & (blockPower2-1)) == 0 && block < s_block.size()) {
      s_block[block] /= 2;
      s_x1[block] += s_block[block];
      s_x2[block] += s_block[block] * s_block[block];

      if (s_block.n_elem-1 <= block) {
        resize(block+2);
      }
      s_block[block+1] += s_block[block];

      s_block[block] = 0;

      block++;
      blockPower2 *= 2;
    }
  }
  void update() override {
    // calculate averages
    MPI_info::barrier();
    MPI_info::reduce_long_long_uint(&n_samples, &total_samples, 1);
    MPI_info::reduce_double(s_x1.memptr(), e_x1.memptr(), s_x1.n_elem);
    MPI_info::reduce_double(s_x2.memptr(), e_x2.memptr(), s_x2.n_elem);

    arma::vec block_sample(s_x1.n_elem);
    for (unsigned long long int block = 0, pow_two_block = 1; block < block_sample.n_elem; block++, pow_two_block <<= 1ull) {
      block_sample[block] = n_samples / (pow_two_block);
    }
    block_sample *= comm_size;

    e_x1  = e_x1 / block_sample;
    e_x2  = e_x2 / block_sample;

    if (0 == master) {
      // calculate variance and derived
      var = e_x2 - (e_x1 % e_x1);
      std = sqrt(var);
      error = sqrt(var / block_sample);
    }
  }
  void to_json(std::string fname) override {

    // call update function
    update();

    if (0 == master) {
    // open stream
      std::ofstream os(fname + ".json");
      // open json
      os << "{\n";

      // print number of steps
      os << "\t\"Steps\" : " << total_samples << ",\n";

      // print E[x]
      os << "\t\"EX\" : [\n";
      for (auto it = e_x1.begin(); it != e_x1.end(); it++) {
        if (it != e_x1.begin()) {
          os << ",\n";
        }
        os << "\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << *it;
      }
      os << "],\n";

      // print Var[x]
      os << "\t\"EX2\" : [\n";
      for (auto it = e_x2.begin(); it != e_x2.end(); it++) {
        if (it != e_x2.begin()) {
          os << ",\n";
        }
        os << "\t\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1) << *it;
      }
      os << "]\n";

      os << "}";
    }
  }
  std::ostream& write(std::ostream& os) override {
    update();
    if (0 == master) {
      os << n_samples << "\t";
      os << std::setprecision(7);
      os << e_x1[0] << "\t";
      for (int block = 0; block < error.n_elem; ++block) {
        os << error[block] << "\t";
      }
    }
    return os;
  }
 private:
  int master;
  int comm_size;
  unsigned long long int n_samples;
  unsigned long long int total_samples;

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

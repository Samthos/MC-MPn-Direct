//
// Created by aedoran on 6/8/18.
//

#ifndef MC_MP3_DIRECT_TAU_INTEGRALS_H
#define MC_MP3_DIRECT_TAU_INTEGRALS_H

#include <algorithm>
#include <iostream>
#include <vector>

#include "basis/qc_basis.h"
#include "qc_random.h"

class Tau {
 public:
  explicit Tau(const Basis& basis) {
    iocc1 = basis.iocc1;
    iocc2 = basis.iocc2;
    ivir1 = basis.ivir1;
    ivir2 = basis.ivir2;

    evals.resize(ivir2);
    std::copy(basis.nw_en, basis.nw_en + ivir2, evals.begin());

    scratch.resize(ivir2);
  }
  virtual void resize(int dimm) = 0;
  virtual void new_tau(Random& random) = 0;
  virtual std::vector<double> get_exp_tau(int, int) = 0;
  virtual double get_gfn_tau(int, int, int ,int) = 0;
  virtual double get_wgt(int) = 0;
  virtual double get_tau(int) = 0;
  virtual bool next() = 0;
  virtual bool is_new(int) = 0;

 protected:
  int iocc1, iocc2, ivir1, ivir2;
  std::vector<double> evals;
  std::vector<double> tau;
  std::vector<double> wgt;
  std::vector<double> scratch;
  std::vector<std::vector<double>> exp_tau;
};

class Stochastic_Tau : public Tau {
 public:
  explicit Stochastic_Tau(const Basis& basis) : Tau(basis) {
    lambda = 2.0 * (evals[ivir1] - evals[iocc2 - 1]);
  }
  ~Stochastic_Tau() = default;

  void resize(int dimm) override {
    tau.resize(dimm);
    wgt.resize(dimm);

    exp_tau.resize(dimm);
    for (auto &it : exp_tau) {
      it.resize(ivir2);
    }
  }
  void new_tau(Random& random) override {
    // generate new tau point and weights
    for (auto i = 0; i < tau.size(); i++) {
      double p = random.uniform();
      tau[i] = -log(1.0 - p) / lambda;
      wgt[i] = 1.0 / (lambda * (1.0 - p));

      for (auto im = iocc1; im < iocc2; im++) {
        exp_tau[i][im] = exp(evals[im] * tau[i]);
      }
      for (auto am = ivir1; am < ivir2; am++) {
        exp_tau[i][am] = exp(-evals[am] * tau[i]);
      }
    }
  }
  std::vector<double> get_exp_tau(int stop, int start) override {
    if (start == stop) {
      return exp_tau[start];
    } else {
      std::fill(scratch.begin(), scratch.end(), 1.0);
      for (auto it = start; it <= stop; it++) {
        std::transform(scratch.begin(), scratch.end(), exp_tau[it].begin(), scratch.begin(), std::multiplies<>());
      }
      return scratch;
    }
  }
  double get_gfn_tau(int stop, int start, int offset, int conjugate) override {
    double s(1.0);
    if (start == stop) {
      s = exp_tau[start][iocc2 + offset];
    } else {
      for (int it = start; it <= stop; it++) {
        s *= exp_tau[it][iocc2 + offset];
      }
    }
    if (conjugate && offset < 0) {
      s = 1.0 / s;
    } else if (!conjugate && offset >= 0) {
      s = 1.0 / s;
    }
    return s;
  }
  double get_wgt(int dimm) override {
    return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<>());
  }
  double get_tau(int index) override {
    return tau[index];
  }
  bool next() override {
    return false;
  }
  bool is_new(int i) override {
    return true;
  }

 private:
  double lambda;
};

class Super_Stochastic_Tau : public Tau {
 public:
  explicit Super_Stochastic_Tau(const Basis& basis) : Tau(basis) {
    set_pdf_cdf(hole_pdf, hole_cdf, iocc1, iocc2);
    set_pdf_cdf(particle_pdf, particle_cdf, ivir1, ivir2);
  }
  ~Super_Stochastic_Tau() = default;

  void resize(int dimm) override {
    lambda.resize(dimm);
    tau.resize(dimm);
    wgt.resize(dimm);

    exp_tau.resize(dimm);
    for (auto &it : exp_tau) {
      it.resize(ivir2);
    }
  }
  void new_tau(Random& random) override {
    // generate new tau point and weights
    for (auto i = 0; i < tau.size(); i++) {
      double p = random.uniform();
      set_lambda(i, random);
      tau[i] = -log(1.0 - p) / lambda[i];
      set_weight(i);

      for (auto im = iocc1; im < iocc2; im++) {
        exp_tau[i][im] = exp(evals[im] * tau[i]);
      }
      for (auto am = ivir1; am < ivir2; am++) {
        exp_tau[i][am] = exp(-evals[am] * tau[i]);
      }
    }
  }
  std::vector<double> get_exp_tau(int stop, int start) override {
    if (start == stop) {
      return exp_tau[start];
    } else {
      std::fill(scratch.begin(), scratch.end(), 1.0);
      for (auto it = start; it <= stop; it++) {
        std::transform(scratch.begin(), scratch.end(), exp_tau[it].begin(), scratch.begin(), std::multiplies<>());
      }
      return scratch;
    }
  }
  double get_gfn_tau(int stop, int start, int offset, int conjugate) override {
    double s(1.0);
    if (start == stop) {
      s = exp_tau[start][iocc2 + offset];
    } else {
      for (int it = start; it <= stop; it++) {
        s *= exp_tau[it][iocc2 + offset];
      }
    }
    if (conjugate && offset < 0) {
      s = 1.0 / s;
    } else if (!conjugate && offset >= 0) {
      s = 1.0 / s;
    }
    return s;
  }
  double get_wgt(int dimm) override {
    return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<>());
  }
  double get_tau(int index) override {
    return tau[index];
  }
  bool next() override {
    return false;
  }
  bool is_new(int i) override {
    return true;
  }

 private:
  void set_pdf_cdf(std::vector<double>& pdf, std::vector<double>& cdf, int first, int last) {
    pdf.resize(last - first);
    cdf.resize(last - first);

    std::transform(evals.begin() + first, evals.begin() + last, pdf.begin(), [](double x) {return std::abs(1/x);});
    auto normalization = 1.0 / std::accumulate(pdf.begin(), pdf.end(), 0.0);
    std::transform(pdf.begin(), pdf.end(), pdf.begin(), [normalization](double x) {return x * normalization;});

    std::partial_sum(pdf.begin(), pdf.end(), cdf.begin());
  }
  void set_lambda(int index, Random& random) {
    auto i = choose_index(random, hole_cdf, iocc1);
    auto j = choose_index(random, hole_cdf, iocc1);
    auto a = choose_index(random, particle_cdf, ivir1);
    auto b = choose_index(random, particle_cdf, ivir1);
    lambda[index] = evals[a] + evals[b] - evals[i] - evals[j];
  }
  static int choose_index(Random& random, std::vector<double>& cdf, int offset) {
    auto p = random.uniform();
    auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
    return std::distance(cdf.begin(), it) + offset;
  }
  void set_weight(int index) {
    wgt[index] = 0.0;
    double hole_weight = 0.0;
    double hole_weight_e = 0.0;
    for (int i = 0; i < hole_pdf.size(); i++) {
      double x = hole_pdf[i] * exp(evals[i+iocc1] * tau[index]);
      hole_weight += x;
      hole_weight_e -= x * evals[i + iocc1];
    }

    double particle_weight = 0.0;
    double particle_weight_e = 0.0;
    for (int i = 0; i < particle_pdf.size(); i++) {
      double x = particle_pdf[i] * exp(-evals[i+ivir1] * tau[index]);
      particle_weight += x;
      particle_weight_e += x * evals[i + ivir1];
    }
    wgt[index] += 2 * hole_weight * hole_weight * particle_weight * particle_weight_e;
    wgt[index] += 2 * particle_weight * particle_weight * hole_weight * hole_weight_e;
    wgt[index] = 1.0 / wgt[index];
  }
  std::vector<double> lambda;
  std::vector<double> hole_pdf, hole_cdf;
  std::vector<double> particle_pdf, particle_cdf;
};

class Quadrature_Tau : public Tau {
 public:
  Quadrature_Tau(const Basis& basis) : Tau(basis) {
    tau = {
        459.528454529921248195023509,
        0.002176143805986910199912,
        75.647524700428292021570087,
        0.013219203192174486943822,
        27.635855710538834273393149,
        0.036184875564343521592292,
        13.821771900816584022209099,
        0.072349623997261858221464,
        8.124825510985218102177896,
        0.123079566280893559770959,
        5.238489369094648573366158,
        0.190894727380696543894700,
        3.574116946388957050118051,
        0.279789389938773946919781,
        2.529798344872996818111233,
        0.395288423690625334572246,
        1.834438449215696431693345,
        0.545125948721552511244681,
        1.349829280916060136874535,
        0.740834425610734315092998,
        1.000000000000000000000000};
    wgt = {
        1240.137264162088286,
        0.005872796340197,
        95.637046659066982,
        0.016712318843111,
        22.450252490071218,
        0.029395130776845,
        8.242542564370559,
        0.043145326054643,
        3.876926315587520,
        0.058729927034166,
        2.128605721895366,
        0.077568088880637,
        1.291883267060868,
        0.101131338617294,
        0.839203153977961,
        0.131127958307208,
        0.573533895185722,
        0.170432651403897,
        0.407885334132820,
        0.223862010922047,
        0.298891108005834};

    exp_tau.resize(tau.size());
    for (auto &it : exp_tau) {
      it.resize(ivir2);

      int i = &it - &exp_tau[0];
      for (auto im = iocc1; im < iocc2; im++) {
        it[im] = exp(evals[im] * tau[i]);
      }
      for (auto am = ivir1; am < ivir2; am++) {
        it[am] = exp(-evals[am] * tau[i]);
      }
    }
  }

  void resize(int dimm) override {
    indices.resize(dimm);
    std::fill(indices.begin(), indices.end(), 0);
  }
  void new_tau(Random& random) override {}
  std::vector<double> get_exp_tau(int stop, int start) override {
    if (start == stop) {
      int index = indices[start];
      return exp_tau[index];
    } else {
      std::fill(scratch.begin(), scratch.end(), 1.0);
      for (auto it = start; it <= stop; it++) {
        int index = indices[it];
        std::transform(scratch.begin(), scratch.end(), exp_tau[index].begin(), scratch.begin(), std::multiplies<>());
      }
      return scratch;
    }
  }
  double get_gfn_tau(int stop, int start, int offset, int conjugate) override {
    double s(1.0);
    if (start == stop) {
      int index = indices[start];
      s = exp_tau[index][iocc2 + offset];
    } else {
      for (int it = start; it <= stop; it++) {
        int index = indices[it];
        s *= exp_tau[index][iocc2 + offset];
      }
    }
    if (conjugate && offset < 0) {
      s = 1.0 / s;
    } else if (!conjugate && offset >= 0) {
      s = 1.0 / s;
    }
    return s;
  }
  double get_wgt(int dimm) override {
    double weight = 1.0;
    for (int i = 0; i < dimm; ++i) {
      weight *= wgt[indices[i]];
    }
    return weight;
  }
  double get_tau(int index) override {
    index = indices[index];
    return tau[index];
  }
  bool next() override {
    auto index = indices.begin();

    (*index)++;
    index++;

    while ((index) != indices.end() && *(index-1) == tau.size()) {
      (*index)++;
      *(index-1) = 0;
      index++;
    }
    if (indices.back() == tau.size()) {
      std::fill(indices.begin(), indices.end(), 0);
      return false;
    }
    return true;
  }
  bool is_new(int i) override {
    return std::all_of(indices.begin() + i, indices.end(), [](int index){return index == 0;});
  }
 private:
  std::vector<int> indices;
};
#endif //MC_MP3_DIRECT_TAU_INTEGRALS_H

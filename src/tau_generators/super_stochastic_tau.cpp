#include <iostream>
#include <algorithm>
#include "super_stochastic_tau.h"

Super_Stochastic_Tau::Super_Stochastic_Tau(const std::shared_ptr<Movec_Parser> basis) : Tau(basis) {
  set_pdf_cdf(hole_pdf, hole_cdf, iocc1, iocc2);
  set_pdf_cdf(particle_pdf, particle_cdf, ivir1, ivir2);
}
size_t Super_Stochastic_Tau::get_n_coordinates() {
  return p.size();
}
void Super_Stochastic_Tau::resize(int dimm) {
  lambda.resize(dimm);
  p.resize(dimm);
  tau.resize(dimm);
  wgt.resize(dimm);

  exp_tau.resize(dimm);
  for (auto &it : exp_tau) {
    it.resize(ivir2);
  }
}
void Super_Stochastic_Tau::new_tau(Random& random) {
  // generate new tau point and weights
  for (auto i = 0; i < tau.size(); i++) {
    p[i] = random.uniform();
    set_lambda(i, random);
    tau[i] = -log(1.0 - p[i]) / lambda[i];
    set_weight(i);

    for (auto im = iocc1; im < iocc2; im++) {
      exp_tau[i][im] = exp(evals[im] * tau[i]);
    }
    for (auto am = ivir1; am < ivir2; am++) {
      exp_tau[i][am] = exp(-evals[am] * tau[i]);
    }
  }
}
std::vector<double> Super_Stochastic_Tau::get_exp_tau(int stop, int start) {
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
double Super_Stochastic_Tau::get_gfn_tau(int stop, int start, int offset, int conjugate) {
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
double Super_Stochastic_Tau::get_wgt(int dimm) {
  return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<>());
}
double Super_Stochastic_Tau::get_tau(int index) {
  return tau[index];
}
bool Super_Stochastic_Tau::next() {
  return false;
}
bool Super_Stochastic_Tau::is_new(int i) {
  return true;
}
void Super_Stochastic_Tau::set_from_other(Tau* other) {
  std::cerr << "Super Stochastic not implemented for dimer\n";
  exit(0);
};
void Super_Stochastic_Tau::set_pdf_cdf(std::vector<double>& pdf, std::vector<double>& cdf, int first, int last) {
  pdf.resize(last - first);
  cdf.resize(last - first);

  std::transform(evals.begin() + first, evals.begin() + last, pdf.begin(), [](double x) {return std::abs(1/(x*x));});
  auto normalization = 1.0 / std::accumulate(pdf.begin(), pdf.end(), 0.0);
  std::transform(pdf.begin(), pdf.end(), pdf.begin(), [normalization](double x) {return x * normalization;});

  std::partial_sum(pdf.begin(), pdf.end(), cdf.begin());
}
void Super_Stochastic_Tau::set_lambda(int index, Random& random) {
  auto i = choose_index(random, hole_cdf, iocc1);
  auto j = choose_index(random, hole_cdf, iocc1);
  auto a = choose_index(random, particle_cdf, ivir1);
  auto b = choose_index(random, particle_cdf, ivir1);
  lambda[index] = evals[a] + evals[b] - evals[i] - evals[j];
}
int Super_Stochastic_Tau::choose_index(Random& random, std::vector<double>& cdf, int offset) {
  auto p = random.uniform();
  auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
  return std::distance(cdf.begin(), it) + offset;
}
void Super_Stochastic_Tau::set_weight(int index) {
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

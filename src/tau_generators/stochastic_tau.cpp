#include <iostream>
#include <algorithm>
#include "stochastic_tau.h"

Stochastic_Tau::Stochastic_Tau(const std::shared_ptr<Movec_Parser> basis) : Tau(basis) {
  lambda = 2.0 * (evals[ivir1] - evals[iocc2 - 1]);
}
void Stochastic_Tau::resize(int dimm) {
  p.resize(dimm);
  tau.resize(dimm);
  wgt.resize(dimm);

  exp_tau.resize(dimm);
  for (auto &it : exp_tau) {
    it.resize(ivir2);
  }
}
size_t Stochastic_Tau::get_n_coordinates() {
  return p.size();
}
void Stochastic_Tau::new_tau(Random& random) {
  // generate new tau point and weights
  for (auto i = 0; i < tau.size(); i++) {
    p[i] = random.uniform();
  }
  update();
}
void Stochastic_Tau::update() {
  // generate new tau point and weights
  for (auto i = 0; i < tau.size(); i++) {
    tau[i] = -log(1.0 - p[i]) / lambda;
    wgt[i] = 1.0 / (lambda * (1.0 - p[i]));

    for (auto im = iocc1; im < iocc2; im++) {
      exp_tau[i][im] = exp(evals[im] * tau[i]);
    }
    for (auto am = ivir1; am < ivir2; am++) {
      exp_tau[i][am] = exp(-evals[am] * tau[i]);
    }
  }
}
std::vector<double> Stochastic_Tau::get_exp_tau(int stop, int start) {
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
double Stochastic_Tau::get_gfn_tau(int stop, int start, int offset, int conjugate) {
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
double Stochastic_Tau::get_wgt(int dimm) {
  return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<>());
}
double Stochastic_Tau::get_tau(int index) {
  return tau[index];
}
bool Stochastic_Tau::next() {
  return false;
}
bool Stochastic_Tau::is_new(int i) {
  return true;
}
void Stochastic_Tau::set_from_other(Tau* other) {
  copy_p(other);
  update();
}
void Stochastic_Tau::set(const std::vector<double>& p_in) {
  if (p_in.size() == p.size()) {
    std::copy(p_in.begin(), p_in.end(), p.begin());
    update();
  } else {
    std::cerr << "In Stochastic_Tau::set input is the not compatable\n";
    exit(0);
  }
}


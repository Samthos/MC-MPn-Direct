#include <iostream>
#include <algorithm>
#include "dummy_tau.h"

Dummy_Tau::Dummy_Tau(const std::shared_ptr<Movec_Parser> basis) : Tau(basis) {
  std::fill(tau.begin(), tau.end(), 2.0);
  std::fill(wgt.begin(), wgt.end(), 2.0);
  for (auto &it : exp_tau) {
    std::iota(it.begin(), it.end(), 1.0);
    std::transform(it.begin(), it.end(), it.begin(), [](double x){return 1.0/x;});
  }
}
void Dummy_Tau::resize(int dimm) {
  p.resize(dimm);
  tau.resize(dimm);
  wgt.resize(dimm);

  exp_tau.resize(dimm);
  for (auto &it : exp_tau) {
    it.resize(ivir2);
  }

  std::fill(tau.begin(), tau.end(), 2.0);
  std::fill(wgt.begin(), wgt.end(), 2.0);
  for (auto &it : exp_tau) {
    std::iota(it.begin(), it.end(), 1.0);
    std::transform(it.begin(), it.end(), it.begin(), [](double x){return 1.0/x;});
  }
}
size_t Dummy_Tau::get_n_coordinates() {
  return p.size();
}
void Dummy_Tau::new_tau(Random& random) { }
std::vector<double> Dummy_Tau::get_exp_tau(int stop, int start) {
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
double Dummy_Tau::get_gfn_tau(int stop, int start, int offset, int conjugate) {
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
double Dummy_Tau::get_wgt(int dimm) {
  return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<>());
}
double Dummy_Tau::get_tau(int index) {
  return tau[index];
}
bool Dummy_Tau::next() {
  return false;
}
bool Dummy_Tau::is_new(int i) {
  return true;
}
void Dummy_Tau::set_from_other(Tau* other) { }

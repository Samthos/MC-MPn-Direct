#include <algorithm>
#include "quadrature_tau.h"

Quadrature_Tau::Quadrature_Tau(const std::shared_ptr<Movec_Parser> basis) : Tau(basis) {
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
void Quadrature_Tau::resize(int dimm) {
  indices.resize(dimm);
  std::fill(indices.begin(), indices.end(), 0);
}
size_t Quadrature_Tau::get_n_coordinates() {
  return indices.size();
}
void Quadrature_Tau::new_tau(Random& random) {}
std::vector<double> Quadrature_Tau::get_exp_tau(int stop, int start) {
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
double Quadrature_Tau::get_gfn_tau(int stop, int start, int offset, int conjugate) {
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
double Quadrature_Tau::get_wgt(int dimm) {
  double weight = 1.0;
  for (int i = 0; i < dimm; ++i) {
    weight *= wgt[indices[i]];
  }
  return weight;
}
double Quadrature_Tau::get_tau(int index) {
  index = indices[index];
  return tau[index];
}
bool Quadrature_Tau::next() {
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
bool Quadrature_Tau::is_new(int i) {
  return std::all_of(indices.begin() + i, indices.end(), [](int index){return index == 0;});
}
void Quadrature_Tau::set_from_other(Tau* other) {
  Quadrature_Tau* o = dynamic_cast<Quadrature_Tau*>(other);
  std::copy(o->indices.begin(), o->indices.end(), indices.begin());
};

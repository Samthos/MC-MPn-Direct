#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "electron_list.h"

std::ostream& operator << (std::ostream& os, const Electron& electron) {
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.pos[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron.weight << ",";
  return os;
}

Electron_List::Electron_List(int size) :
    electrons(size),
    pos(size),
    weight(size),
    inverse_weight(size) {}
void Electron_List::set_weight(Electron& electron, const Electron_GTO_Weight& weight) {
  electron.weight = weight.weight(electron.pos);
  electron.inverse_weight = 1.0 / electron.weight;
}
void Electron_List::transpose() {
  for (size_t i = 0; i < electrons.size(); i++) {
    pos[i] = electrons[i].pos;
    weight[i] = electrons[i].weight;
    inverse_weight[i] = electrons[i].inverse_weight;
  }
}

Electron_List* create_electron_sampler(IOPs& iops, Molecule& molec, Electron_GTO_Weight& weight) {
  Electron_List* electron_list = nullptr;
  if (iops.iopns[KEYS::SAMPLER] == SAMPLER::DIRECT) {
    electron_list = new Direct_Electron_List(iops.iopns[KEYS::ELECTRONS]);
  } else if (iops.iopns[KEYS::SAMPLER] == SAMPLER::METROPOLIS) {
    std::string str = iops.sopns[KEYS::SEED_FILE];
    if (!str.empty()) {
      str += ".electron_list_metropolis";
    }
    Random rnd(iops.iopns[KEYS::DEBUG], str);
    electron_list = new Metropolis_Electron_List(iops.iopns[KEYS::ELECTRONS], iops.dopns[KEYS::MC_DELX], rnd, molec, weight);
  }
  return electron_list;
}

bool Direct_Electron_List::requires_blocking() {
  return false;
}
void Direct_Electron_List::move(Random& random, const Electron_GTO_Weight& weight) {
  for (Electron &electron : electrons) {
    mc_move_scheme(electron, random, weight);
  }
  transpose();
}
void Direct_Electron_List::mc_move_scheme(Electron& electron, Random& random, const Electron_GTO_Weight& weight) {
  // choose function to sample;
  double rnd = random.uniform();
  auto it = std::lower_bound(std::begin(weight.cum_sum), std::end(weight.cum_sum), rnd);
  auto index = static_cast<int>(std::distance(weight.cum_sum.begin(), it));
  auto atom = weight.cum_sum_index[index][0];
  auto prim = weight.cum_sum_index[index][1];

  // compute some parameters
  double alpha = 1.0 / sqrt(2.0 * weight.mcBasisList[atom].alpha[prim]);

  // sample x, y, and theta
  for (int i = 0; i < 3; i++) {
    electron.pos[i] = random.normal(weight.mcBasisList[atom].center[i], alpha);
  }

  set_weight(electron, weight);
}

Metropolis_Electron_List::Metropolis_Electron_List(int size, double ml, Random& random, const Molecule& molec, const Electron_GTO_Weight& weight) : Electron_List(size),
    move_length(ml),
    moves_since_rescale(0),
    successful_moves(0),
    failed_moves(0)
{
  // initilizie pos
  for (Electron& electron : electrons) {
    initialize(electron, random, molec, weight);
  }
  // burn in
  for (int i = 0; i < 100'000; i++) {
    move(random, weight);
  }
}
bool Metropolis_Electron_List::requires_blocking() {
  return true;
}
void Metropolis_Electron_List::move(Random& random, const Electron_GTO_Weight& weight) {
  if (moves_since_rescale == 1'000) {
    rescale_move_length();
  }
  for (Electron &electron : electrons) {
    mc_move_scheme(electron, random, weight);
  }
  moves_since_rescale++;
  transpose();
}
void Metropolis_Electron_List::initialize(Electron &electron, Random &random, const Molecule &molec, const Electron_GTO_Weight& weight) {
  int atom;
  double amp1, amp2, theta1, theta2;
  std::array<double, 3> pos;
  constexpr double twopi = 6.283185307179586;

  atom = molec.atoms.size() * random.uniform();
  pos[0] = molec.atoms[atom].pos[0];
  pos[1] = molec.atoms[atom].pos[1];
  pos[2] = molec.atoms[atom].pos[2];

  amp1 = sqrt(-0.5 * log(random.uniform() * 0.2));
  amp2 = sqrt(-0.5 * log(random.uniform() * 0.5));
  theta1 = twopi * random.uniform();
  theta2 = 0.5 * twopi * random.uniform();

  electron.pos[0] = pos[0] + amp1*cos(theta1);
  electron.pos[1] = pos[1] + amp1*sin(theta1);
  electron.pos[2] = pos[2] + amp2*cos(theta2);

  set_weight(electron, weight);
}
void Metropolis_Electron_List::mc_move_scheme(Electron &electron, Random &random, const Electron_GTO_Weight &weight) {
  Electron trial_electron = electron;

  for (int i = 0; i < 3; i++) {
    trial_electron.pos[i] += random.uniform(-move_length, move_length);
  }

  set_weight(trial_electron, weight);

  auto ratio = trial_electron.weight / electron.weight;

  auto rval = random.uniform(0, 1);
  if (rval < 1.0E-3) {
    rval = 1.0E-3;
  }

  if (ratio > rval) {
    std::swap(trial_electron, electron);
    successful_moves++;
  } else {
    failed_moves++;
  }
}
void Metropolis_Electron_List::rescale_move_length() {
  double ratio = ((double) failed_moves)/((double) (failed_moves + successful_moves));
  if (ratio < 0.5) {
    ratio = std::min(1.0/(2.0*ratio), 1.1);
  } else {
    ratio = std::max(0.9, 1.0/(2.0*ratio));
  }
  move_length = move_length * ratio;
  moves_since_rescale = 0;
  successful_moves = 0;
  failed_moves = 0;
}

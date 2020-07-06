#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "metropolis_electron_pair_list.h"

Metropolis_Electron_Pair_List::Metropolis_Electron_Pair_List(int size, double ml, Random& random, const Molec& molec, const Electron_Pair_GTO_Weight& weight) : Electron_Pair_List(size),
    move_length(ml),
    moves_since_rescale(0),
    successful_moves(0),
    failed_moves(0)
{
  // initilizie pos
  for (Electron_Pair& electron_pair : electron_pairs) {
    initialize(electron_pair, random, molec, weight);
  }
  // burn in
  for (int i = 0; i < 100'000; i++) {
    move(random, weight);
  }
}
bool Metropolis_Electron_Pair_List::requires_blocking() {
  return true;
}
void Metropolis_Electron_Pair_List::move(Random& random, const Electron_Pair_GTO_Weight& weight) {
  if (moves_since_rescale == 1'000) {
    rescale_move_length();
  }
  for (Electron_Pair &electron_pair : electron_pairs) {
    mc_move_scheme(electron_pair, random, weight);
  }
  moves_since_rescale++;
  transpose();
}
void Metropolis_Electron_Pair_List::initialize(Electron_Pair &electron_pair, Random &random, const Molec &molec, const Electron_Pair_GTO_Weight& weight) {
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

  electron_pair.pos1[0] = pos[0] + amp1*cos(theta1);
  electron_pair.pos1[1] = pos[1] + amp1*sin(theta1);
  electron_pair.pos1[2] = pos[2] + amp2*cos(theta2);

  //elec position 2;
  atom = molec.atoms.size() * random.uniform();
  pos[0] = molec.atoms[atom].pos[0];
  pos[1] = molec.atoms[atom].pos[1];
  pos[2] = molec.atoms[atom].pos[2];

  amp1 = sqrt(-0.5 * log(random.uniform() * 0.2));
  amp2 = sqrt(-0.5 * log(random.uniform() * 0.5));
  theta1 = twopi * random.uniform();
  theta2 = 0.5 * twopi * random.uniform();

  electron_pair.pos2[0] = pos[0] + amp1*cos(theta1);
  electron_pair.pos2[1] = pos[1] + amp1*sin(theta1);
  electron_pair.pos2[2] = pos[2] + amp2*cos(theta2);

  set_weight(electron_pair, weight);
}
void Metropolis_Electron_Pair_List::mc_move_scheme(Electron_Pair &electron_pair, Random &random, const Electron_Pair_GTO_Weight &weight) {
  Electron_Pair trial_electron_pair = electron_pair;

  for (int i = 0; i < 3; i++) {
    trial_electron_pair.pos1[i] += random.uniform(-move_length, move_length);
    trial_electron_pair.pos2[i] += random.uniform(-move_length, move_length);
  }

  set_weight(trial_electron_pair, weight);

  auto ratio = trial_electron_pair.wgt / electron_pair.wgt;

  auto rval = random.uniform(0, 1);
  if (rval < 1.0E-3) {
    rval = 1.0E-3;
  }

  if (ratio > rval) {
    std::swap(trial_electron_pair, electron_pair);
    successful_moves++;
  } else {
    failed_moves++;
  }
}
void Metropolis_Electron_Pair_List::rescale_move_length() {
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

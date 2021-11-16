#include "metropolis_electron_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Metropolis_Electron_List<Container, Allocator>::Metropolis_Electron_List(int size, double ml, Random& random, const Molecule& molec, const Electron_GTO_Weight& weight) : Electron_List<Container, Allocator>(size),
    move_length(ml),
    moves_since_rescale(0),
    successful_moves(0),
    failed_moves(0)
{
  // initilizie pos
  for (Electron& electron : this->electrons) {
    initialize(electron, random, molec, weight);
  }
  // burn in
  for (int i = 0; i < 100'000; i++) {
    move(random, weight);
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Metropolis_Electron_List<Container, Allocator>::requires_blocking() {
  return true;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Metropolis_Electron_List<Container, Allocator>::move(Random& random, const Electron_GTO_Weight& weight) {
  if (moves_since_rescale == 1'000) {
    rescale_move_length();
  }
  for (Electron &electron : this->electrons) {
    mc_move_scheme(electron, random, weight);
  }
  moves_since_rescale++;
  this->transpose();
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Metropolis_Electron_List<Container, Allocator>::initialize(Electron &electron, Random &random, const Molecule &molec, const Electron_GTO_Weight& weight) {
  int atom;
  double amp1, amp2, theta1, theta2;
  Point pos;
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

  Electron_List<Container, Allocator>::set_weight(electron, weight);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Metropolis_Electron_List<Container, Allocator>::mc_move_scheme(Electron &electron, Random &random, const Electron_GTO_Weight &weight) {
  Electron trial_electron = electron;

  for (int i = 0; i < 3; i++) {
    trial_electron.pos[i] += random.uniform(-move_length, move_length);
  }

  Electron_List<Container, Allocator>::set_weight(trial_electron, weight);

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

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Metropolis_Electron_List<Container, Allocator>::rescale_move_length() {
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

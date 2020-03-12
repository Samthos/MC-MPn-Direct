#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "electron_pair_list.h"

std::ostream& operator << (std::ostream& os, const Electron_Pair& electron_pair) {
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos1[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[0] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[1] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.pos2[2] << ",";
  os << std::setprecision(std::numeric_limits<double>::digits10 + 1) << electron_pair.wgt << ",";
  return os;
}

Electron_Pair_List::Electron_Pair_List(int size) :
    electron_pairs(size),
    pos1(size),
    pos2(size),
    wgt(size),
    rv(size),
    r12(size) {}
double Electron_Pair_List::calculate_r12(const Electron_Pair &electron_pair_list) {
  double r12;
  std::array<double, 3> dr{};
  std::transform(electron_pair_list.pos1.begin(), electron_pair_list.pos1.end(), electron_pair_list.pos2.begin(), dr.begin(),
                 std::minus<>());
  r12 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
  return sqrt(r12);
}
void Electron_Pair_List::set_weight(Electron_Pair& electron_pair, const Electron_Pair_GTO_Weight& weight) {
  electron_pair.wgt = weight.weight(electron_pair.pos1, electron_pair.pos2);
  electron_pair.r12 = calculate_r12(electron_pair);
  electron_pair.rv = 1.0 / (electron_pair.r12 * electron_pair.wgt);
}
void Electron_Pair_List::transpose() {
  for (size_t i = 0; i < electron_pairs.size(); i++) {
    pos1[i] = electron_pairs[i].pos1;
    pos2[i] = electron_pairs[i].pos2;
    wgt[i] = electron_pairs[i].wgt;
    rv[i] = electron_pairs[i].rv;
    r12[i] = electron_pairs[i].r12;
  }
}

Electron_Pair_List* create_electron_pair_sampler(IOPs& iops, Molec& molec, Electron_Pair_GTO_Weight& weight) {
  Electron_Pair_List* electron_pair_list = nullptr;
  if (iops.iopns[KEYS::SAMPLER] == SAMPLER::DIRECT) {
    electron_pair_list = new Direct_Electron_Pair_List(iops.iopns[KEYS::ELECTRON_PAIRS]);
  } else if (iops.iopns[KEYS::SAMPLER] == SAMPLER::METROPOLIS) {
    Random rnd(iops.iopns[KEYS::DEBUG]);
    electron_pair_list = new Metropolis_Electron_Pair_List(iops.iopns[KEYS::ELECTRON_PAIRS], iops.dopns[KEYS::MC_DELX], rnd, molec, weight);
  }
  return electron_pair_list;
}

bool Direct_Electron_Pair_List::requires_blocking() {
  return false;
}
void Direct_Electron_Pair_List::move(Random& random, const Electron_Pair_GTO_Weight& weight) {
  for (Electron_Pair &electron_pair : electron_pairs) {
    mc_move_scheme(electron_pair, random, weight);
  }
  transpose();
}
void Direct_Electron_Pair_List::mc_move_scheme(Electron_Pair& electron_pair, Random& random, const Electron_Pair_GTO_Weight& weight) {
  constexpr double TWOPI = 6.283185307179586;
  std::array<double, 3> dr{};

  // choose function to sample;
  double rnd = random.uniform();
  auto it = std::lower_bound(std::begin(weight.cum_sum),
                   std::end(weight.cum_sum), rnd);
  auto index = static_cast<int>(std::distance(weight.cum_sum.begin(), it));
  auto prim2 = weight.cum_sum_index[index][3];
  auto prim1 = weight.cum_sum_index[index][2];
  auto a2 = weight.cum_sum_index[index][1];
  auto a1 = weight.cum_sum_index[index][0];

  // compute some parameters
  double alpha = weight.mcBasisList[a1].alpha[prim1];
  double beta  = weight.mcBasisList[a2].alpha[prim2];
  double gamma = 1.0/sqrt(2.0*(alpha+beta));

  // sample x, y, and theta
  double x = random.normal(0.0, gamma);
  double y = random.normal(0.0, gamma);
  double theta = random.uniform(0.0, TWOPI); // 2*pi

  double z, r, phi;
  if (a1 != a2) {
    // compute distance bewteen the centers
    std::transform(weight.mcBasisList[a1].center.begin(), weight.mcBasisList[a1].center.end(),
                   weight.mcBasisList[a2].center.begin(), dr.begin(),
                   std::minus<>());
    double rab = sqrt(std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0));

    // sample z coordinate
    double mu_z = rab * (alpha - beta) / (2.0*(alpha + beta));
    z = random.normal(mu_z, gamma);
    r = calculate_r(random, alpha, beta, rab);
    phi = calculate_phi(random.uniform(), r, alpha, beta, rab);
  } else {
    z = random.normal(0.0, gamma);
    r = 2.0 * sqrt(-alpha * beta * log(1.0- random.uniform()) / (alpha+beta));
    phi = acos(1.0-2.0*random.uniform());
  }

  electron_pair.pos1[0] = x - r * cos(theta) * sin(phi) / (2.0 * alpha);
  electron_pair.pos1[1] = y - r * sin(theta) * sin(phi) / (2.0 * alpha);
  electron_pair.pos1[2] = z - r * cos(phi) / (2.0 * alpha);
  electron_pair.pos2[0] = x + r * cos(theta) * sin(phi) / (2.0 * beta);
  electron_pair.pos2[1] = y + r * sin(theta) * sin(phi) / (2.0 * beta);
  electron_pair.pos2[2] = z + r * cos(phi) / (2.0 * beta);

  // compute center of the two gaussians
  std::transform(weight.mcBasisList[a1].center.begin(), weight.mcBasisList[a1].center.end(),
                 weight.mcBasisList[a2].center.begin(), dr.begin(),
                 std::plus<>());
  std::for_each(dr.begin(), dr.end(), [](double&x){x/=2;});

  // if centers are not then same, then rotate
  if (a1 != a2) {
    std::array<double, 3> rb(weight.mcBasisList[a1].center);

    std::transform(rb.begin(), rb.end(), dr.begin(), rb.begin(), std::minus<>());
    double rb_norm = sqrt(std::inner_product(rb.begin(), rb.end(), rb.begin(), 0.0));

    double r_p = acos(rb[2]/rb_norm);
    double r_t = atan2(rb[1], rb[0]);

    x = electron_pair.pos1[0]; z = electron_pair.pos1[2];
    electron_pair.pos1[0] = z * sin(r_p) + x * cos(r_p);
    electron_pair.pos1[2] = z * cos(r_p) - x * sin(r_p);
    x = electron_pair.pos2[0]; z = electron_pair.pos2[2];
    electron_pair.pos2[0] = z * sin(r_p) + x * cos(r_p);
    electron_pair.pos2[2] = z * cos(r_p) - x * sin(r_p);

    x = electron_pair.pos1[0]; y = electron_pair.pos1[1];
    electron_pair.pos1[0] = x * cos(r_t) - y * sin(r_t);
    electron_pair.pos1[1] = x * sin(r_t) + y * cos(r_t);
    x = electron_pair.pos2[0]; y = electron_pair.pos2[1];
    electron_pair.pos2[0] = x * cos(r_t) - y * sin(r_t);
    electron_pair.pos2[1] = x * sin(r_t) + y * cos(r_t);
  }

  // shift result to original center
  std::transform(electron_pair.pos1.begin(), electron_pair.pos1.end(), dr.begin(), electron_pair.pos1.begin(), std::plus<>());
  std::transform(electron_pair.pos2.begin(), electron_pair.pos2.end(), dr.begin(), electron_pair.pos2.begin(), std::plus<>());

  set_weight(electron_pair, weight);
}
double Direct_Electron_Pair_List::CDF(const double& rho, const double& c, const double& erf_c) {
  return (2.0 * erf_c + erf(rho - c) - erf(rho + c)) / (2.0 * erf_c);
}
double Direct_Electron_Pair_List::PDF(const double& rho, const double& c, const double& erf_c) {
  constexpr double sqrt_pi = 1.772453850905516;
  double rhopc = rho + c;
  double rhomc = rho - c;
  double exp_rhopc = exp(-rhopc * rhopc);
  double exp_rhomc = exp(-rhomc * rhomc);
  return (exp_rhomc - exp_rhopc) / (sqrt_pi * erf_c);
}
double Direct_Electron_Pair_List::PDF_Prime(const double& rho, const double& c, const double& erf_c) {
  constexpr double sqrt_pi = 1.772453850905516;
  double rhopc = rho + c;
  double rhomc = rho - c;
  double exp_rhopc = exp(-rhopc * rhopc);
  double exp_rhomc = exp(-rhomc * rhomc);
  return 2.0 * (rhopc * exp_rhopc - rhomc * exp_rhomc) / (sqrt_pi * erf_c);
}
double Direct_Electron_Pair_List::calculate_r(Random& random, double alpha, double beta, double a) {
  constexpr double sqrt_pi = 1.772453850905516;
  auto gamma = sqrt(alpha * beta / ( alpha + beta));
  auto c = a * gamma;
  auto erf_c = erf(c);

  // guess rho
  double rho, p;
  if (c < 1.25) {
    // generate guess from r exp(-alpha^2 r^2) (2 alpha^2)
    // where alpha == sqrt[pi] erc[c] / (2 c)
    p = random.uniform();
    rho = 2 * c * sqrt(-log(1-p)) / (sqrt_pi * erf_c);
  } else {
    // guess from standard normal
    rho = random.normal(c, 1 / sqrt(2));
    p = (1 - erf(c - rho)) / 2;
    // relfect if rho less than zero
    if (rho < 0) {
      rho *= -1;
    }
  }

  // calculate itital cdf
  double cdf, pdf, pdf_prime;

  // iterate until cdf - p is less than 10E-6
  int iter = 0;
  do {
    cdf = CDF(rho, c, erf_c) - p;
    pdf = PDF(rho, c, erf_c);
    pdf_prime = PDF_Prime(rho, c, erf_c);
    rho = rho - 2.0 * cdf * pdf / (2.0*pdf*pdf-cdf*pdf_prime);
    iter++;
  } while (std::abs(cdf) > 10E-6);
  return rho * 2.0 * gamma;
}
double Direct_Electron_Pair_List::calculate_phi(double p, double r, double alpha, double beta, double a) {
  constexpr double sqrt_pi = 1.772453850905516;
  auto gamma = sqrt(alpha * beta / ( alpha + beta));
  auto c = a * gamma;
  auto rho = r / (2.0 * gamma);
  p = p * PDF(rho, c, erf(c)) / (2.0 * gamma);

  auto phi = log(exp(-(c+rho)*(c+rho)) + 2.0 * p * sqrt_pi * gamma * erf(c));
  phi = - 2 * c * rho / (c*c + rho*rho + phi);
  phi = acos(1.0/phi);
  return phi;
}

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

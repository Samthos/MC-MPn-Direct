#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "direct_electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Direct_Electron_Pair_List<Container, Allocator>::Direct_Electron_Pair_List(int size) : Electron_Pair_List<Container, Allocator>(size) {}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Direct_Electron_Pair_List<Container, Allocator>::requires_blocking() {
  return false;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Direct_Electron_Pair_List<Container, Allocator>::move(Random& random, const Electron_Pair_GTO_Weight& weight) {
  for (Electron_Pair &electron_pair : this->electron_pairs) {
    mc_move_scheme(electron_pair, random, weight);
  }
  this->transpose();
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Direct_Electron_Pair_List<Container, Allocator>::mc_move_scheme(Electron_Pair& electron_pair, Random& random, const Electron_Pair_GTO_Weight& weight) {
  constexpr double TWOPI = 6.283185307179586;
  Point dr;

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
    dr = weight.mcBasisList[a1].center - weight.mcBasisList[a2].center;
    double rab = dr.length();

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
  dr = weight.mcBasisList[a1].center + weight.mcBasisList[a2].center;
  dr *= 0.5;

  // if centers are not then same, then rotate
  if (a1 != a2) {
    Point rb = weight.mcBasisList[a1].center - dr;
    double rb_norm = rb.length();

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
  electron_pair.pos1 += dr;
  electron_pair.pos2 += dr;

  Electron_Pair_List<Container, Allocator>::set_weight(electron_pair, weight);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double Direct_Electron_Pair_List<Container, Allocator>::CDF(const double& rho, const double& c, const double& erf_c) {
  return (2.0 * erf_c + erf(rho - c) - erf(rho + c)) / (2.0 * erf_c);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double Direct_Electron_Pair_List<Container, Allocator>::PDF(const double& rho, const double& c, const double& erf_c) {
  constexpr double sqrt_pi = 1.772453850905516;
  double rhopc = rho + c;
  double rhomc = rho - c;
  double exp_rhopc = exp(-rhopc * rhopc);
  double exp_rhomc = exp(-rhomc * rhomc);
  return (exp_rhomc - exp_rhopc) / (sqrt_pi * erf_c);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double Direct_Electron_Pair_List<Container, Allocator>::PDF_Prime(const double& rho, const double& c, const double& erf_c) {
  constexpr double sqrt_pi = 1.772453850905516;
  double rhopc = rho + c;
  double rhomc = rho - c;
  double exp_rhopc = exp(-rhopc * rhopc);
  double exp_rhomc = exp(-rhomc * rhomc);
  return 2.0 * (rhopc * exp_rhopc - rhomc * exp_rhomc) / (sqrt_pi * erf_c);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double Direct_Electron_Pair_List<Container, Allocator>::calculate_r(Random& random, double alpha, double beta, double a) {
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

template <template <typename, typename> typename Container, template <typename> typename Allocator>
double Direct_Electron_Pair_List<Container, Allocator>::calculate_phi(double p, double r, double alpha, double beta, double a) {
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

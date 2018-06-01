#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <functional>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "el_pair.h"
#include "qc_constant.h"

void el_pair_typ::init(const int ivir2) {
  psi1.resize(ivir2);
  psi2.resize(ivir2);
}

void el_pair_typ::pos_init(const Molec& molec, Random& rand) {
  int i;
  double amp1, amp2, theta1, theta2;
  double pos[3];

  // elec position 1
  i = static_cast<int>(molec.atom.size() * rand.get_rand());
  pos[0] = molec.atom[i].pos[0];
  pos[1] = molec.atom[i].pos[1];
  pos[2] = molec.atom[i].pos[2];

  amp1 = sqrt(-0.5 * log(rand.get_rand() * 0.2));
  amp2 = sqrt(-0.5 * log(rand.get_rand() * 0.5));
  theta1 = twopi * rand.get_rand();
  theta2 = pi * rand.get_rand();

  pos1[0] = pos[0] + amp1 * cos(theta1);
  pos1[1] = pos[1] + amp1 * sin(theta1);
  pos1[2] = pos[2] + amp2 * cos(theta2);

  // elec position 2;
  i = static_cast<int>(molec.atom.size() * rand.get_rand());
  pos[0] = molec.atom[i].pos[0];
  pos[1] = molec.atom[i].pos[1];
  pos[2] = molec.atom[i].pos[2];

  amp1 = sqrt(-0.5 * log(rand.get_rand() * 0.2));
  amp2 = sqrt(-0.5 * log(rand.get_rand() * 0.5));
  theta1 = twopi * rand.get_rand();
  theta2 = pi * rand.get_rand();

  pos2[0] = pos[0] + amp1 * cos(theta1);
  pos2[1] = pos[1] + amp1 * sin(theta1);
  pos2[2] = pos[2] + amp2 * cos(theta2);
}

double el_pair_typ::r12() {
  double r12;
  std::array<double, 3> dr{};
  std::transform(pos1.begin(), pos1.end(), pos2.begin(), dr.begin(),
                 std::minus<double>());
  r12 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
  return sqrt(r12);
}

void el_pair_typ::mc_move_scheme(Random& rand,
                                 const Molec& molec,
                                 const GTO_Weight& mc_basis) {
#ifndef NDEBUG
  static int count = 0;
  static int step = 0;
  count = (count+1);
  if (count == 32) {
    count=0;
    step++;
  }
#endif
  std::array<double, 3> dr{};

  // choose function to sample;
  double rnd = rand.uniform(0.0, 1.0);
  auto it = std::lower_bound(std::begin(mc_basis.cum_sum),
                   std::end(mc_basis.cum_sum), rnd);
  auto index = static_cast<int>(std::distance(mc_basis.cum_sum.begin(), it));
  auto prim2 = mc_basis.cum_sum_index[index][3];
  auto prim1 = mc_basis.cum_sum_index[index][2];
  auto a2 = mc_basis.cum_sum_index[index][1];
  auto a1 = mc_basis.cum_sum_index[index][0];

  // compute some parameters
  double alpha = mc_basis.mcBasisList[a1].alpha[prim1];
  double beta  = mc_basis.mcBasisList[a2].alpha[prim2];
  double gamma = 1.0/sqrt(2.0*(alpha+beta));

  // sample x, y, and theta
  double x = rand.normal(0.0, gamma);
  double y = rand.normal(0.0, gamma);
  double theta = rand.uniform(0.0, twopi);

  double z, r, phi;
  if (a1 != a2) {
    // compute distance bewteen the centers
    std::transform(mc_basis.mcBasisList[a1].center.begin(), mc_basis.mcBasisList[a1].center.end(),
                   mc_basis.mcBasisList[a2].center.begin(), dr.begin(),
                   std::minus<double>());
    double rab = sqrt(std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0));

    // sample z coordinate
    double mu_z = rab * (alpha - beta) / (2.0*(alpha + beta));
    z = rand.normal(mu_z, gamma);
    r = calculate_r(rand.uniform(0.0, 1.0), alpha, beta, rab);
    phi = calculate_phi(rand.uniform(0.0, 1.0), r, alpha, beta, rab);
  } else {
    z = rand.normal(0.0, gamma);
    r = 2.0 * sqrt(-alpha * beta * log(1.0-rand.get_rand()) / (alpha+beta));
    phi = acos(1.0-2.0*rand.uniform(0.0, 1.0));
  }

  pos1[0] = x - r * cos(theta) * sin(phi) / (2.0 * alpha);
  pos1[1] = y - r * sin(theta) * sin(phi) / (2.0 * alpha);
  pos1[2] = z - r * cos(phi) / (2.0 * alpha);
  pos2[0] = x + r * cos(theta) * sin(phi) / (2.0 * beta);
  pos2[1] = y + r * sin(theta) * sin(phi) / (2.0 * beta);
  pos2[2] = z + r * cos(phi) / (2.0 * beta);

  // compute center of the two gaussians
  std::transform(mc_basis.mcBasisList[a1].center.begin(), mc_basis.mcBasisList[a1].center.end(),
                 mc_basis.mcBasisList[a2].center.begin(), dr.begin(),
                 std::plus<double>());
  std::for_each(dr.begin(), dr.end(), [](double&x){x/=2;});

  // if centers are not then same, then rotate
  if (a1 != a2) {
    std::array<double, 3> rb(mc_basis.mcBasisList[a1].center);

    std::transform(rb.begin(), rb.end(), dr.begin(), rb.begin(), std::minus<double>());
    double rb_norm = sqrt(std::inner_product(rb.begin(), rb.end(), rb.begin(), 0.0));

    double r_p = acos(rb[2]/rb_norm);
    double r_t = atan2(rb[1], rb[0]);

    x = pos1[0]; z = pos1[2];
    pos1[0] = z * sin(r_p) + x * cos(r_p);
    pos1[2] = z * cos(r_p) - x * sin(r_p);
    x = pos2[0]; z = pos2[2];
    pos2[0] = z * sin(r_p) + x * cos(r_p);
    pos2[2] = z * cos(r_p) - x * sin(r_p);

    x = pos1[0]; y = pos1[1];
    pos1[0] = x * cos(r_t) - y * sin(r_t);
    pos1[1] = x * sin(r_t) + y * cos(r_t);
    x = pos2[0]; y = pos2[1];
    pos2[0] = x * cos(r_t) - y * sin(r_t);
    pos2[1] = x * sin(r_t) + y * cos(r_t);
  }

  // shift result to original center
  std::transform(pos1.begin(), pos1.end(), dr.begin(), pos1.begin(), std::plus<double>());
  std::transform(pos2.begin(), pos2.end(), dr.begin(), pos2.begin(), std::plus<double>());

  wgt = mc_basis.weight(pos1, pos2);
  rv = 1.0 / (r12()*wgt);
  is_new = true;

#ifndef NDEBUG
  std::cout << "pos";
  std::cout << "," << step ;
  std::cout << "," << count;
  std::cout << ",\"";
  for (auto &it : mc_basis.cum_sum_index[index]) {
    std::cout << it;
  }
  std::cout << "\"";
  std::cout << "," << pos1[0];
  std::cout << "," << pos1[1];
  std::cout << "," << pos1[2];
  std::cout << "," << pos2[0];
  std::cout << "," << pos2[1];
  std::cout << "," << pos2[2];
  std::cout << "," << r12();
  /*
  std::cout << "," << x;
  std::cout << "," << y;
  std::cout << "," << z;
  */
  std::cout << ","  << r;
  std::cout << ","  << phi;
  /*
  std::cout << theta << " ";
  */
  std::cout << "," << wgt << std::endl;
#endif  // NDEBUG
}

double CDF(const double& rho, const double& c, const double& erf_c) {
  return (2.0 * erf_c + erf(rho - c) - erf(rho + c)) / (2.0 * erf_c);
}
double PDF(const double& rho, const double& c, const double& erf_c) {
  return exp(-(c+rho)*(c+rho)) * (exp(4*c*rho)-1.0) / (sqrt_pi * erf_c);
}
double PDF_Prime(const double& rho, const double& c, const double& erf_c) {
  return 2.0 * exp(-(c+rho)*(c+rho)) * (c+rho + (c-rho)*exp(4*c*rho)) / (sqrt_pi * erf_c);
}
double el_pair_typ::calculate_r(double p, double alpha, double beta, double a) {
  auto gamma = sqrt(alpha * beta / ( alpha + beta));
  auto c = a * gamma;
  auto erf_c = erf(c);

  // guess rho
  auto rho = c / erf_c;

  // calculate itital cdf
  double cdf = CDF(rho, c, erf_c) - p;
  double pdf, pdf_prime;

  // iterate until cdf - p is less than 10E-6
  while (std::abs(cdf) > 10E-6) {
    cdf = CDF(rho, c, erf_c) - p;
    pdf = PDF(rho, c, erf_c);
    pdf_prime = PDF_Prime(rho, c, erf_c);
    rho = rho - 2.0 * cdf * pdf / (2.0*pdf*pdf-cdf*pdf_prime);
  }
  return rho * 2.0 * gamma;
}
double el_pair_typ::calculate_phi(double p, double r, double alpha, double beta, double a) {
  auto gamma = sqrt(alpha * beta / ( alpha + beta));
  auto c = a * gamma;
  auto rho = r / (2.0 * gamma);
  p = p * PDF(rho, c, erf(c)) / (2.0 * gamma);

  auto phi = log(exp(-(c+rho)*(c+rho)) + 2.0 * p * sqrt_pi * gamma * erf(c));
  phi = - 2 * c * rho / (c*c + rho*rho + phi);
  phi = acos(1.0/phi);
  return phi;
}

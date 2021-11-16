//
// Created by aedoran on 12/18/19.
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "correlation_factor_data.h"
#include "correlation_factor_function.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Correlation_Factor_Data<Container, Allocator>::Correlation_Factor_Data(int electrons_in,
    int electron_pairs_in,
    CORRELATION_FACTOR::Type correlation_factor_in,
    double gamma_in,
    double beta_in) :
    f12p(electron_pairs_in),
    f12p_a(electron_pairs_in),
    f12p_c(electron_pairs_in),
    f12o(electrons_in * electrons_in, 0.0),
    f12o_b(electrons_in * electrons_in, 0.0),
    f12o_d(electrons_in * electrons_in, 0.0),
    f13(electron_pairs_in * electrons_in, 0.0),
    f23(electron_pairs_in * electrons_in, 0.0),
    correlation_factor(correlation_factor_in) 
{
  auto cf = create_correlation_factor(correlation_factor, gamma_in, beta_in);
  m_f12d_is_zero = cf->f12_d_is_zero();
  gamma = cf->gamma();
  beta  = cf->beta();
  delete cf;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
bool Correlation_Factor_Data<Container, Allocator>::f12_d_is_zero() {
  return m_f12d_is_zero;
}

Correlation_Factor_Data_Host::Correlation_Factor_Data_Host(int electrons_in,
    int electron_pairs_in,
    CORRELATION_FACTOR::Type correlation_factor_in,
    double gamma_in,
    double beta_in) :
    Correlation_Factor_Data(electrons_in, electron_pairs_in, correlation_factor_in, gamma_in, beta_in),
    m_correlation_factor(create_correlation_factor(correlation_factor_in, gamma_in, beta_in)) {}

void Correlation_Factor_Data_Host::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = m_correlation_factor->f12(electron_pair_list->r12[ip]);
    f12p_a[ip] = m_correlation_factor->f12_a(electron_pair_list->r12[ip]);
    f12p_c[ip] = m_correlation_factor->f12_c(electron_pair_list->r12[ip]);
  }

  for (int io = 0; io < electron_list->size(); io++) {
    for (int jo = 0; jo < electron_list->size(); jo++) {
      if (jo != io) {
        auto dr = Point::distance(electron_list->pos[io], electron_list->pos[jo]);
        f12o[io * electron_list->size() + jo] = m_correlation_factor->f12(dr);
        f12o_b[io * electron_list->size() + jo] = m_correlation_factor->f12_b(dr);
        f12o_d[io * electron_list->size() + jo] = m_correlation_factor->f12_d(dr);
      } else {
        f12o[io * electron_list->size() + jo] = 0.0;
        f12o_b[io * electron_list->size() + jo] = 0.0;
        f12o_d[io * electron_list->size() + jo] = 0.0;
      }
    }
  }

  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for (int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = m_correlation_factor->f12(Point::distance(electron_pair_list->pos1[ip], electron_list->pos[io]));
      f23[ip * electron_list->size() + io] = m_correlation_factor->f12(Point::distance(electron_pair_list->pos2[ip], electron_list->pos[io]));
    }
  }
}

Slater_Correlation_Factor_Data_Host::Slater_Correlation_Factor_Data_Host(int electrons_in,
    int electron_pairs_in,
    double gamma_in,
    double beta_in) :
    Correlation_Factor_Data(electrons_in, electron_pairs_in, CORRELATION_FACTOR::Slater, gamma_in, beta_in) {}

void Slater_Correlation_Factor_Data_Host::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = Slater_f12(electron_pair_list->r12[ip], gamma, beta);
    f12p_a[ip] = 2.0 * gamma * f12p[ip];
    f12p_c[ip] = -gamma * f12p[ip];
  }

  for (int io = 0, idx = 0; io < electron_list->size(); io++) {
    for (int jo = 0; jo < electron_list->size(); jo++, idx++) {
      if (jo != io) {
        f12o[idx] = Slater_f12(Point::distance(electron_list->pos[io], electron_list->pos[jo]), gamma, beta);
        f12o_b[idx] = -gamma * gamma * f12o[idx];
      } else {
        f12o[idx] = 0.0;
        f12o_b[idx] = 0.0;
      }
    }
  }

  for (int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for (int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = Slater_f12(Point::distance(electron_pair_list->pos1[ip], electron_list->pos[io]), gamma, beta);
      f23[ip * electron_list->size() + io] = Slater_f12(Point::distance(electron_pair_list->pos2[ip], electron_list->pos[io]), gamma, beta);
    }
  }
}

template <>
Correlation_Factor_Data<std::vector, std::allocator>* create_Correlation_Factor_Data<std::vector, std::allocator>(int electrons,
    int electron_pairs,
    CORRELATION_FACTOR::Type correlation_factor,
    double gamma,
    double beta) {
  Correlation_Factor_Data<std::vector, std::allocator>* correlation_factor_data;
  switch (correlation_factor) {
    case CORRELATION_FACTOR::Slater: correlation_factor_data =  new Slater_Correlation_Factor_Data_Host(electrons, electron_pairs, gamma, beta); break;
    default:
      correlation_factor_data = new Correlation_Factor_Data_Host(electrons, electron_pairs, correlation_factor, gamma, beta);
      break;
  }
  return correlation_factor_data;
}

#ifdef HAVE_CUDA
template <>
Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>* create_Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>(int electrons,
    int electron_pairs,
    CORRELATION_FACTOR::Type correlation_factor,
    double gamma,
    double beta) {
  Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>* correlation_factor_data;
  switch (correlation_factor) {
    case CORRELATION_FACTOR::Linear: correlation_factor_data =  new Linear_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Rational: correlation_factor_data =  new Rational_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Slater: correlation_factor_data =  new Slater_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Slater_Linear: correlation_factor_data =  new Slater_Linear_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Gaussian: correlation_factor_data =  new Gaussian_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Cusped_Gaussian: correlation_factor_data =  new Cusped_Gaussian_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Yukawa_Coulomb: correlation_factor_data =  new Yukawa_Coulomb_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Jastrow: correlation_factor_data =  new Jastrow_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::ERFC: correlation_factor_data =  new ERFC_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::ERFC_Linear: correlation_factor_data =  new ERFC_Linear_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Tanh: correlation_factor_data =  new Tanh_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::ArcTan: correlation_factor_data =  new ArcTan_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Logarithm: correlation_factor_data =  new Logarithm_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Hybrid: correlation_factor_data =  new Hybrid_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Two_Parameter_Rational: correlation_factor_data =  new Two_Parameter_Rational_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Higher_Rational: correlation_factor_data =  new Higher_Rational_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Cubic_Slater: correlation_factor_data =  new Cubic_Slater_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
    case CORRELATION_FACTOR::Higher_Jastrow: correlation_factor_data =  new Higher_Jastrow_Correlation_Factor_Data_Device(electrons, electron_pairs, gamma, beta);  break;
  }
  return correlation_factor_data;
}
#endif

#define SOURCE_FILE "correlation_factor_data.imp.cpp"
#include "correlation_factor_patterns.h"
#undef SOURCE_FILE

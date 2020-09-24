//
// Created by aedoran on 12/18/19.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "correlation_factor_data.h"

Correlation_Factor_Data::Correlation_Factor_Data(int electrons_in, 
      int electron_pairs_in, 
      CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_in,
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
    correlation_factor(correlation_factor_in),
    gamma(gamma_in),
    beta(beta_in)
{
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor, gamma, beta);
  m_f12d_is_zero = m_correlation_factor->f12_d_is_zero();
}

void Correlation_Factor_Data::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor, gamma, beta);
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = m_correlation_factor->f12(electron_pair_list->r12[ip]);
    f12p_a[ip] = m_correlation_factor->f12_a(electron_pair_list->r12[ip]);
    f12p_c[ip] = m_correlation_factor->f12_c(electron_pair_list->r12[ip]);
  }
  
  for(int io = 0; io < electron_list->size();io++) {
    for(int jo = 0; jo < electron_list->size();jo++) {
      if (jo != io) {
        auto dr = Point::distance(electron_list->pos[io], electron_list->pos[jo]);
        f12o[io * electron_list->size() + jo]  = m_correlation_factor->f12(dr);
        f12o_b[io * electron_list->size() + jo] =  m_correlation_factor->f12_b(dr);
        f12o_d[io * electron_list->size() + jo] =  m_correlation_factor->f12_d(dr);
      } else {
        f12o[io * electron_list->size() + jo]   = 0.0;
        f12o_b[io * electron_list->size() + jo] = 0.0;
        f12o_d[io * electron_list->size() + jo] = 0.0;
      }
    }
  }

  for(int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for(int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = m_correlation_factor->f12(Point::distance(electron_pair_list->pos1[ip], electron_list->pos[io]));
      f23[ip * electron_list->size() + io] = m_correlation_factor->f12(Point::distance(electron_pair_list->pos2[ip], electron_list->pos[io]));
    }
  }
}

bool Correlation_Factor_Data::f12_d_is_zero() {
  return m_f12d_is_zero;
}

/*
void Slater_Correlation_Factor_Data::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  for (int ip = 0; ip < electron_pair_list->size(); ip++) {
    f12p[ip] = f12(electron_pair_list->r12[ip]);
    f12p_a[ip] = 2.0 * gamma * f12p[ip];
    f12p_c[ip] = -gamma * f12p[ip];
  }
  
  for(int io = 0, idx = 0; io < electron_list->size(); io++) {
    for(int jo = 0; jo < electron_list->size(); jo++, idx++) {
      if (jo != io) {
        f12o[idx] = f12(Point::distance(electron_list->pos[io], electron_list->pos[jo]));
        f12o_b[idx] =  -gamma * gamma * f12o[idx];
      } else {
        f12o[idx]   = 0.0;
        f12o_b[idx] = 0.0;
      }
    }
  }

  for(int ip = 0; ip < electron_pair_list->size(); ++ip) {
    for(int io = 0; io < electron_list->size(); ++io) {
      f13[ip * electron_list->size() + io] = f12(Point::distance(electron_pair_list->pos1[ip], electron_list->pos[io]));
      f23[ip * electron_list->size() + io] = f12(Point::distance(electron_pair_list->pos2[ip], electron_list->pos[io]));
    }
  }
}
*/


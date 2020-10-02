//
// Created by aedoran on 12/18/19.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "correlation_factor_data.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
Correlation_Factor_Data<Container, Allocator>::Correlation_Factor_Data(int electrons_in, 
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
  delete m_correlation_factor;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
bool Correlation_Factor_Data<Container, Allocator>::f12_d_is_zero() {
  return m_f12d_is_zero;
}

template <>
void Correlation_Factor_Data<std::vector, std::allocator>::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
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

  delete m_correlation_factor;
}

#ifdef HAVE_CUDA
__global__ 
void f12p_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const double* r12, double* f12p) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p[tid] = m_correlation_factor->f12(r12[tid]);
  }
  delete m_correlation_factor;
}

__global__ 
void f12p_a_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const double* r12, double* f12p_a) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p_a[tid] = m_correlation_factor->f12_a(r12[tid]);
  }
  delete m_correlation_factor;
}

__global__ 
void f12p_c_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const double* r12, double* f12p_c) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    f12p_c[tid] = m_correlation_factor->f12_c(r12[tid]);
  }
  delete m_correlation_factor;
}

__global__ 
void f12o_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const Point* pos, double* f12o) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o[tid] = m_correlation_factor->f12(dr);
  }
  delete m_correlation_factor;
}

__global__ 
void f12o_b_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const Point* pos, double* f12o_b) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o_b[tid] = m_correlation_factor->f12_b(dr);
  }
  delete m_correlation_factor;
}

__global__ 
void f12o_d_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int size, const Point* pos, double* f12o_d) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * size + tidx;
  if (tidx < size && tidy < size && tidx != tidy) {
    auto dr = Point::distance(pos[tidx], pos[tidy]);
    f12o_d[tid] = m_correlation_factor->f12_d(dr);
  }
  delete m_correlation_factor;
}

__global__ 
void f13_kernal(
    CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_id, double gamma, double beta, 
    int electron_pairs, const Point* electron_pair_pos,
    int electrons, const Point* electron_pos, double* f13) {
  auto m_correlation_factor = create_correlation_factor_function(correlation_factor_id, gamma, beta);
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = tidy * electrons + tidx;
  if (tidx < electrons && tidy < electron_pairs) {
    auto dr = Point::distance(electron_pair_pos[tidy], electron_pos[tidx]);
    f13[tid] = m_correlation_factor->f12(dr);
  }
  delete m_correlation_factor;
}

template <>
void Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>::update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((electron_pair_list->size() + 127) / 128, 1, 1);
  f12p_kernal  <<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p.data().get());
  f12p_a_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p_a.data().get());
  f12p_c_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_pair_list->size(), electron_pair_list->r12.data().get(), f12p_c.data().get());

  block_size = dim3(16, 16, 1);
  grid_size = dim3((electron_list->size() + 15) / 16, (electron_list->size() + 15) / 16, 1);
  f12o_kernal  <<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o.data().get());
  f12o_b_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o_b.data().get());
f12o_d_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, electron_list->size(), electron_list->pos.data().get(), f12o_d.data().get());

  block_size = dim3(16, 16, 1);
  grid_size = dim3((electron_list->size() + 15) / 16, (electron_pair_list->size() + 15) / 16,  1);
  f13_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, 
      electron_pair_list->size(), electron_pair_list->pos1.data().get(), 
      electron_list->size(), electron_list->pos.data().get(), 
      f13.data().get());
  f13_kernal<<<grid_size, block_size>>>(correlation_factor, gamma, beta, 
      electron_pair_list->size(), electron_pair_list->pos2.data().get(), 
      electron_list->size(), electron_list->pos.data().get(), 
      f23.data().get());
}
#endif

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


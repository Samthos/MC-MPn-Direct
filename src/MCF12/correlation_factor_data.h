//
// Created by aedoran on 12/31/19.
//

#ifndef CORRELATION_FACTORS_DATA_H_
#define CORRELATION_FACTORS_DATA_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#endif

#include <memory>
#include <vector>

#include "correlation_factor.h"
#include "correlation_factor_types.h"
#include "electron_list.h"
#include "electron_pair_list.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Correlation_Factor_Data {
 protected:
  typedef Container<double, Allocator<double>> vector_double;
  typedef Electron_Pair_List<Container, Allocator> Electron_Pair_List_Type;
  typedef Electron_List<Container, Allocator> Electron_List_Type;

 public:
  Correlation_Factor_Data(int electrons_in,
      int electron_pairs,
      CORRELATION_FACTOR::Type correlation_factor_in,
      double gamma_in,
      double beta_in);
  virtual void update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) = 0;
  bool f12_d_is_zero();

  // electron_pair arrays
  vector_double f12p;
  vector_double f12p_a;
  vector_double f12p_c;

  // electron electron arrays
  vector_double f12o;
  vector_double f12o_b;
  vector_double f12o_d;

  // electron_pair electron arrays
  vector_double f13;
  vector_double f23;

 protected:
  // std::shared_ptr<Correlation_Factor_Function> m_correlation_factor;
  CORRELATION_FACTOR::Type correlation_factor;
  double gamma;
  double beta;
  bool m_f12d_is_zero;
};

template <template <typename, typename> typename Container, template <typename> typename Allocator>
Correlation_Factor_Data<Container, Allocator>* create_Correlation_Factor_Data(int electrons_in,
    int electron_pairs,
    CORRELATION_FACTOR::Type correlation_factor_in,
    double gamma_in,
    double beta_in) {
  printf("Default create mp2 function. Returns nullptr\n");
  exit(0);
  return nullptr;
}

template class Correlation_Factor_Data<std::vector, std::allocator>;

class Correlation_Factor_Data_Host : public Correlation_Factor_Data<std::vector, std::allocator> {
 public:
  Correlation_Factor_Data_Host(int electrons_in,
      int electron_pairs_in,
      CORRELATION_FACTOR::Type correlation_factor_in,
      double gamma_in,
      double beta_in);
  void update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) override;

 protected:
  std::shared_ptr<Correlation_Factor> m_correlation_factor;
};

class Slater_Correlation_Factor_Data_Host : public Correlation_Factor_Data<std::vector, std::allocator> {
 public:
  Slater_Correlation_Factor_Data_Host(int electrons_in,
      int electron_pairs_in,
      double gamma_in,
      double beta_in);
  void update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list) override;
};

template <>
Correlation_Factor_Data<std::vector, std::allocator>* create_Correlation_Factor_Data<std::vector, std::allocator>(int electrons_in,
    int electron_pairs,
    CORRELATION_FACTOR::Type correlation_factor_in,
    double gamma_in,
    double beta_in);

#ifdef HAVE_CUDA
template class Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>;

template <>
Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>* create_Correlation_Factor_Data<thrust::device_vector, thrust::device_allocator>(int electrons_in,
    int electron_pairs,
    CORRELATION_FACTOR::Type correlation_factor_in,
    double gamma_in,
    double beta_in);
#endif

#define SOURCE_FILE "correlation_factor_data.imp.h"
#include "correlation_factor_patterns.h"
#undef SOURCE_FILE

#endif  //  CORRELATION_FACTORS_DATA_H_

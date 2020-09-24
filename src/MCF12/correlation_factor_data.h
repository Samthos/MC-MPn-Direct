//
// Created by aedoran on 12/31/19.
//

#ifndef CORRELATION_FACTORS_DATA_H_
#define CORRELATION_FACTORS_DATA_H_

#include <memory>
#include <vector>

#include "electron_list.h"
#include "electron_pair_list.h"
#include "correlation_factor_function.h"

class Correlation_Factor_Data {
 protected:
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List_Host Electron_List_Type;
  typedef std::vector<double> vector_double;

 public:
  Correlation_Factor_Data(int electrons_in, 
      int electron_pairs, 
      CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor_in,
      double gamma_in,
      double beta_in);
  virtual void update(const Electron_Pair_List_Type* electron_pair_list, const Electron_List_Type* electron_list);
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

 private:
  // std::shared_ptr<Correlation_Factor_Function> m_correlation_factor;
  CORRELATION_FACTORS::CORRELATION_FACTORS correlation_factor;
  double gamma;
  double beta;
  bool m_f12d_is_zero;
};
#endif //  CORRELATION_FACTORS_DATA_H_

//
// Created by aedoran on 12/31/19.
//

#ifndef CORRELATION_FACTORS_H_
#define CORRELATION_FACTORS_H_

#include <memory>
#include <vector>

#include "../qc_input.h"
#include "electron_list.h"
#include "electron_pair_list.h"
#include "correlation_factor_function.h"

class Correlation_Factor_Data {
 protected:
  typedef Electron_Pair_List_Host Electron_Pair_List_Type;
  typedef Electron_List_Host Electron_List_Type;
  typedef std::vector<double> vector_double;

 public:
  Correlation_Factor_Data(const IOPs& iops);
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
  std::shared_ptr<Correlation_Factor_Function> m_correlation_factor;
};
#endif //  CORRELATION_FACTORS_H_

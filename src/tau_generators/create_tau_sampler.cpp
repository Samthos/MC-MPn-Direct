#include "tau.h"
#include "quadrature_tau.h"
#include "stochastic_tau.h"
#include "super_stochastic_tau.h"

Tau* create_tau_sampler(const TAU_GENERATORS::TAU_GENERATORS& tau_generator, const std::shared_ptr<Movec_Parser> basis) {
  Tau* tau;
  switch (tau_generator) {
    case TAU_GENERATORS::STOCHASTIC:  tau = new Stochastic_Tau(basis); break;
    case TAU_GENERATORS::SUPER_STOCH: tau = new Super_Stochastic_Tau(basis); break;
    case TAU_GENERATORS::QUADRATURE:  tau = new Quadrature_Tau(basis); break;
  }
  return tau;
}

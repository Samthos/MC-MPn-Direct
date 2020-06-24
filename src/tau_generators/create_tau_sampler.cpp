#include "tau.h"
#include "quadrature_tau.h"
#include "stochastic_tau.h"
#include "super_stochastic_tau.h"

Tau* create_tau_sampler(const IOPs& iops, const std::shared_ptr<Movec_Parser> basis) {
  Tau* tau;
  switch (iops.iopns[KEYS::TAU_INTEGRATION]) {
    case TAU_INTEGRATION::STOCHASTIC:  tau = new Stochastic_Tau(basis); break;
    case TAU_INTEGRATION::SUPER_STOCH: tau = new Super_Stochastic_Tau(basis); break;
    case TAU_INTEGRATION::QUADRATURE:  tau = new Quadrature_Tau(basis); break;
  }
  return tau;
}

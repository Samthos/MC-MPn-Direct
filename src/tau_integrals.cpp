
#include "qc_input.h"
#include "basis/qc_basis.h"
#include "tau_integrals.h"

Tau* create_tau_sampler(const IOPs& iops, const Basis& basis) {
  Tau* tau;
  switch (iops.iopns[KEYS::TAU_INTEGRATION]) {
    case TAU_INTEGRATION::STOCHASTIC:  tau = new Stochastic_Tau(basis); break;
    case TAU_INTEGRATION::SUPER_STOCH: tau = new Super_Stochastic_Tau(basis); break;
    case TAU_INTEGRATION::QUADRATURE:  tau = new Quadrature_Tau(basis); break;
  }
  return tau;
}

#ifndef TAU_GENERATORS_H_
#define TAU_GENERATORS_H_

namespace TAU_GENERATORS {
  enum TAU_GENERATORS {
    STOCHASTIC,
    QUADRATURE,
    SUPER_STOCH,
  };
  const std::vector<std::string> tau_integration_strings = {
    "STOCHASTIC",
    "QUADRATURE",
    "SUPER_STOCH"
  };
}

#endif // TAU_GENERATORS_H_

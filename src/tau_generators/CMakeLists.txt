set(source
  tau.h
  tau.cpp
  create_tau_sampler.cpp
  quadrature_tau.h
  quadrature_tau.cpp
  stochastic_tau.h
  stochastic_tau.cpp
  super_stochastic_tau.h
  super_stochastic_tau.cpp
)

add_library(tau STATIC ${source})

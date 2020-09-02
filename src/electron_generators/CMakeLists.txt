set(source
  create_electron_pair_list.cpp
  direct_electron_pair_list.cpp
  direct_electron_pair_list.h
  dummy_electron_pair_list.h
  electron_list.cpp
  electron_list.h
  electron_pair_list.cpp
  electron_pair_list.h
  metropolis_electron_pair_list.cpp
  metropolis_electron_pair_list.h
  samplers.h
  weight_function.cpp
  weight_function.h
)

add_library(electron_generators STATIC ${source})
target_include_directories(electron_generators PRIVATE ../geometry)


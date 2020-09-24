set(source
  direct_electron_list.cpp
  direct_electron_pair_list.cpp
  dummy_electron_list.cpp
  dummy_electron_pair_list.cpp
  electron_list.cpp
  electron_pair_list.cpp
  metropolis_electron_list.cpp
  metropolis_electron_pair_list.cpp
  weight_function.cpp
)
# direct_electron_pair_list.h
# dummy_electron_pair_list.h
# electron_list.h
# electron_pair_list.h
# metropolis_electron_pair_list.h
# samplers.h
# weight_function.h

add_library(electron_generators STATIC ${source})
target_include_directories(electron_generators PRIVATE ../geometry)
if(${Enable_CUDA})
  set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(electron_generators PRIVATE -DHAVE_CUDA)
endif()


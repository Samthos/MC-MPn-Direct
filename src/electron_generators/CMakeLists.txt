set(electron_generator_source
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
add_library(electron_generators STATIC ${electron_generator_source})
target_include_directories(electron_generators PRIVATE ../geometry)
if(${Enable_CUDA})
  set_source_files_properties(${electron_generator_source} PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(electron_generators PRIVATE -DHAVE_CUDA)
endif()


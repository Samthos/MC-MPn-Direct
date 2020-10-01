set(correlation_factor_data_source
  correlation_factor_data_test.cpp
  ../../src/MCF12/correlation_factor.cpp
  ../../src/MCF12/correlation_factor_data.cpp
  ../../src/MCF12/correlation_factor_function.cpp
)
add_executable(correlation_factor_data_test.x 
  ${correlation_factor_data_source}
)
target_link_libraries(correlation_factor_data_test.x gtest_main)
target_link_libraries(correlation_factor_data_test.x geometry)
target_link_libraries(correlation_factor_data_test.x electron_generators)
target_link_libraries(correlation_factor_data_test.x ${BLAS_LIBRARIES})
target_link_libraries(correlation_factor_data_test.x blas_wrapper)
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/geometry)
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/electron_generators)
target_include_directories(correlation_factor_data_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/blas_wrapper)
if (${Enable_CUDA})
  set_target_properties(correlation_factor_data_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(correlation_factor_data_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(correlation_factor_data_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(${correlation_factor_data_source} PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME correlation_factor_data_test COMMAND correlation_factor_data_test.x)

set(f12_traces_source
  f12_traces_test.cpp
  ../../src/MCF12/F12_Traces.cpp
  ../../src/basis/wavefunction.cpp
  ../../src/basis/dummy_movec_parser.cpp
)
add_executable(f12_traces_test.x 
  ${f12_traces_source}
)
target_link_libraries(f12_traces_test.x ${BLAS_LIBRARIES})
target_link_libraries(f12_traces_test.x blas_wrapper)
target_link_libraries(f12_traces_test.x electron_generators)
target_link_libraries(f12_traces_test.x geometry)
target_link_libraries(f12_traces_test.x gtest_main)
target_include_directories(f12_traces_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(f12_traces_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(f12_traces_test.x PRIVATE ../../src/electron_generators)
target_include_directories(f12_traces_test.x PRIVATE ../../src/geometry)
target_include_directories(f12_traces_test.x PRIVATE ../../src/basis)
if (${Enable_CUDA})
  set_target_properties(f12_traces_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(f12_traces_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(f12_traces_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(${f12_traces_source} PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME f12_traces_test COMMAND correlation_factor_data_test.x)

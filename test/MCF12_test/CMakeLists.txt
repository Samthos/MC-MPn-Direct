set(correlation_factor_data_source
  correlation_factor_data_test.cpp
  ../../src/MCF12/correlation_factor.cpp
  ../../src/MCF12/correlation_factor_data.cpp
  ../../src/MCF12/correlation_factor_function.cpp
)
add_executable(correlation_factor_data_test.x 
  ${correlation_factor_data_source}
)
set_target_properties(correlation_factor_data_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(correlation_factor_data_test.x gtest_main)
target_link_libraries(correlation_factor_data_test.x geometry)
target_link_libraries(correlation_factor_data_test.x electron_generators)
target_link_libraries(correlation_factor_data_test.x ${BLAS_LIBRARIES})
target_link_libraries(correlation_factor_data_test.x ${CUBLAS_LIBRARIES})
target_link_libraries(correlation_factor_data_test.x blas_wrapper)
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/geometry)
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/electron_generators)
target_include_directories(correlation_factor_data_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(correlation_factor_data_test.x PRIVATE ../../src/blas_wrapper)
if (${Enable_CUDA})
  target_compile_definitions(correlation_factor_data_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(
    ${correlation_factor_data_source}
    PROPERTIES LANGUAGE CUDA)
endif()

add_test(NAME mp2_funcational_test COMMAND correlation_factor_data_test.x)

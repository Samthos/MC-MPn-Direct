add_executable(blas_calls_test.x 
  blas_calls_test.cpp
)
target_link_libraries(blas_calls_test.x gtest_main)
target_link_libraries(blas_calls_test.x blas_wrapper)
target_include_directories(blas_calls_test.x PRIVATE ../../src/blas_wrapper)
add_test(NAME blas_calls_test COMMAND blas_calls_test.x)



set(source 
  blas_wrapper_test.cpp
)

add_executable(blas_wrapper_test.x 
  ${source}
)
if(${Enable_CUDA})
  set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(blas_wrapper_test.x PRIVATE -DHAVE_CUDA)
endif()
target_link_libraries(blas_wrapper_test.x gtest_main)
target_link_libraries(blas_wrapper_test.x blas_wrapper)
target_include_directories(blas_wrapper_test.x PRIVATE ../../src/blas_wrapper)
add_test(NAME blas_wrapper_test COMMAND blas_wrapper_test.x)

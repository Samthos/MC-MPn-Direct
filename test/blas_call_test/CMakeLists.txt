add_executable(blas_calls_test.x 
  blas_calls_test.cpp
  ../../src/blas_calls.h
  ../../src/blas_calls.cpp
)
target_link_libraries(blas_calls_test.x gtest_main)
add_test(NAME blas_calls_test COMMAND blas_calls_test.x)

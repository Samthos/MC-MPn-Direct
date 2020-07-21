#cuda_add_executable(atomic_orbital_test.x 
#  atomic_orbital_test.cu
#)
#target_link_libraries(ovps_set_test.x gtest_main)
#target_link_libraries(ovps_set_test.x ${BLAS_LIBRARIES})
#target_include_directories(ovps_set_test.x PUBLIC ${BLAS_INCLUDE_DIRS})
#target_compile_definitions(ovps_set_test.x PUBLIC -DHAVE_CUDA)
#cuda_add_cublas_to_target(ovps_set_test.x)
#add_test(NAME ATOMIC_ORBITAL_TEST COMMAND atomic_orbital_test.x)

add_executable(shell_test.x
  shell_test.cpp
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
)
target_link_libraries(shell_test.x gtest_main)
add_test(NAME SHELL_TEST COMMAND shell_test.x)

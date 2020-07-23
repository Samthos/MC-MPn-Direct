add_executable(shell_test.x
  shell_test.cpp
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
)
target_link_libraries(shell_test.x gtest_main)
add_test(NAME SHELL_TEST COMMAND shell_test.x)

cuda_add_executable(atomic_orbital_test.x 
  atomic_orbital_test.cu
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
  ../../src/basis/atomic_orbital.h
  ../../src/basis/atomic_orbital.cu
)
target_link_libraries(atomic_orbital_test.x gtest_main)
target_compile_definitions(atomic_orbital_test.x PUBLIC -DHAVE_CUDA)
add_test(NAME ATOMIC_ORBITAL_TEST COMMAND atomic_orbital_test.x)

add_executable(shell_test.x
  shell_test.cpp
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
)
target_link_libraries(shell_test.x gtest_main)
add_test(NAME SHELL_TEST COMMAND shell_test.x)

add_executable(atomic_orbital_test.x 
  atomic_orbital_test.cu
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
  ../../src/basis/atomic_orbital.h
  ../../src/basis/atomic_orbital.cu
)
set_target_properties(atomic_orbital_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(atomic_orbital_test.x gtest_main)
target_link_libraries(atomic_orbital_test.x geometry)
target_include_directories(atomic_orbital_test.x PRIVATE ../../src/geometry)
target_compile_definitions(atomic_orbital_test.x PRIVATE -DHAVE_CUDA)
add_test(NAME ATOMIC_ORBITAL_TEST COMMAND atomic_orbital_test.x)

add_executable(basis_test.x 
  basis_test.cu
  ../../src/qc_mpi.h
  ../../src/qc_mpi.cpp
  ../../src/basis/dummy_basis_parser.h
  ../../src/basis/dummy_basis_parser.cpp
  ../../src/basis/basis_parser.h
  ../../src/basis/basis_parser.cpp
  ../../src/basis/shell.h
  ../../src/basis/shell.cpp
  ../../src/basis/basis.h
  ../../src/basis/basis.cu
  ../../src/basis/atomic_orbital.h
  ../../src/basis/atomic_orbital.cu
)
set_target_properties(basis_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(basis_test.x gtest_main)
target_link_libraries(basis_test.x ${BLAS_LIBRARIES})
target_link_libraries(basis_test.x geometry)
target_include_directories(basis_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(basis_test.x PRIVATE ../../src/geometry)
target_compile_definitions(basis_test.x PRIVATE -DHAVE_CUDA)
add_test(NAME BASIS_TEST COMMAND basis_test.x)

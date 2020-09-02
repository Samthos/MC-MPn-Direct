add_executable(mp2_functional_test.x 
  mp2_functional_test.cu
  ../../src/MCMP/mp_functional.h
  ../../src/MCMP/mp2_functional.h
  ../../src/MCMP/mp2_functional.cpp
  ../../src/MCMP/direct_mp2_functional.h
  ../../src/MCMP/direct_mp2_functional.cpp
  ../../src/MCMP/device_mp2_functional.h
  ../../src/MCMP/device_mp2_functional.cu
  ../../src/MCMP/create_mp2_functional.h
  ../../src/MCMP/create_mp2_functional.cpp
  ../../src/basis/dummy_movec_parser.cpp
  ../../src/basis/dummy_movec_parser.h
  ../../src/basis/movec_parser.cpp
  ../../src/basis/movec_parser.h
  ../../src/basis/wavefunction.cu
  ../../src/basis/wavefunction.h
  ../../src/electron_generators/dummy_electron_pair_list.h
  ../../src/electron_generators/electron_pair_list.cpp
  ../../src/electron_generators/electron_pair_list.h
  ../../src/blas_calls.cpp
  ../../src/blas_calls.h
  ../../src/ovps_set.cu
  ../../src/ovps_set.h
  ../../src/qc_mpi.cpp
  ../../src/qc_mpi.h
  ../../src/qc_ovps.cu
  ../../src/qc_ovps.h
  ../../src/qc_random.cpp
  ../../src/qc_random.h
)
set_target_properties(mp2_functional_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(mp2_functional_test.x gtest_main)
target_link_libraries(mp2_functional_test.x geometry)
target_link_libraries(mp2_functional_test.x tau)
target_link_libraries(mp2_functional_test.x ${BLAS_LIBRARIES})
target_link_libraries(mp2_functional_test.x ${CUBLAS_LIBRARIES})
target_include_directories(mp2_functional_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(mp2_functional_test.x PRIVATE ../../src/geometry)
target_include_directories(mp2_functional_test.x PRIVATE ../../src/tau_generators)
target_include_directories(mp2_functional_test.x PRIVATE ../../src/electron_generators)
target_compile_definitions(mp2_functional_test.x PRIVATE -DHAVE_CUDA)
add_test(NAME mp2_funcational_test COMMAND mp2_functional_test.x)
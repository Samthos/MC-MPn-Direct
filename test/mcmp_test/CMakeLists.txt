set(mp2_functional_test_source
  mp2_functional_test.cpp
  ../../src/MCMP/create_mp2_functional.cpp
  ../../src/MCMP/device_mp2_functional.cu
  ../../src/MCMP/direct_mp2_functional.cpp
  ../../src/MCMP/mp2_functional.cpp
  ../../src/basis/dummy_movec_parser.cpp
  ../../src/basis/movec_parser.cpp
  ../../src/basis/wavefunction.cpp
  ../../src/ovps_set.cpp
  ../../src/qc_mpi.cpp
  ../../src/qc_ovps.cpp
  ../../src/qc_random.cpp)
add_executable(mp2_functional_test.x ${mp2_functional_test_source})
target_link_libraries(mp2_functional_test.x ${BLAS_LIBRARIES})
target_link_libraries(mp2_functional_test.x blas_wrapper)
target_link_libraries(mp2_functional_test.x electron_generators)
target_link_libraries(mp2_functional_test.x geometry)
target_link_libraries(mp2_functional_test.x gtest_main)
target_link_libraries(mp2_functional_test.x test_helper)
target_link_libraries(mp2_functional_test.x tau)
target_include_directories(mp2_functional_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(mp2_functional_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(mp2_functional_test.x PRIVATE ../../src/electron_generators)
target_include_directories(mp2_functional_test.x PRIVATE ../../src/geometry)
target_include_directories(mp2_functional_test.x PRIVATE ../../src/tau_generators)
target_include_directories(mp2_functional_test.x PRIVATE ../test_helper)
target_compile_definitions(mp2_functional_test.x PRIVATE -DHAVE_CUDA)
if (${Enable_CUDA})
  set_target_properties(mp2_functional_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(mp2_functional_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(mp2_functional_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(${mp2_functional_test_source} PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME mp2_funcational_test COMMAND mp2_functional_test.x)

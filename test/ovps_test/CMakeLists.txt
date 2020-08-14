message("ovps test " ${gtest_SOURCE_DIR}/)

add_executable(ovps_set_test.x 
  ovps_set_test.cu
  ovps_test_helper.h
  ../../src/ovps_set.cu
  ../../src/ovps_set.h
  ../../src/blas_calls.h
  ../../src/blas_calls.cpp
)
set_target_properties(ovps_set_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ovps_set_test.x gtest_main)
target_link_libraries(ovps_set_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_set_test.x ${CUBLAS_LIBRARIES})
target_include_directories(ovps_set_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_compile_definitions(ovps_set_test.x PRIVATE -DHAVE_CUDA)
add_test(NAME OVPS_SET_TEST COMMAND ovps_set_test.x)


add_executable(ovps_test.x 
  ovps_test.cu
  ovps_test_helper.h
  ../../src/qc_ovps.h
  ../../src/qc_ovps.cu
  ../../src/ovps_set.h
  ../../src/ovps_set.cu
  ../../src/qc_mpi.h
  ../../src/qc_mpi.cpp
  ../../src/basis/wavefunction.h
  ../../src/basis/wavefunction.cpp
  ../../src/basis/movec_parser.h
  ../../src/basis/movec_parser.cpp
  ../../src/basis/dummy_movec_parser.h
  ../../src/basis/dummy_movec_parser.cpp
  ../../src/qc_random.h
  ../../src/qc_random.cpp
  ../../src/blas_calls.h
  ../../src/blas_calls.cpp
)
set_target_properties(ovps_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ovps_test.x gtest_main)
target_link_libraries(ovps_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_test.x ${CUBLAS_LIBRARIES})
target_link_libraries(ovps_test.x tau)
target_include_directories(ovps_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(ovps_test.x PRIVATE ../../src/tau_generators)
target_include_directories(ovps_test.x PRIVATE ../../src/geometry)
target_compile_definitions(ovps_test.x PRIVATE -DHAVE_CUDA)
add_test(NAME OVPS_TEST COMMAND ovps_test.x)

message("ovps test " ${gtest_SOURCE_DIR}/)

add_executable(ovps_set_test.x 
  ovps_set_test.cu
  ../../src/ovps_set.cpp
)
target_link_libraries(ovps_set_test.x gtest_main)
target_link_libraries(ovps_set_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_set_test.x blas_wrapper)
target_include_directories(ovps_set_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(ovps_set_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
if (${Enable_CUDA})
  set_target_properties(ovps_set_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(ovps_set_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(ovps_set_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(
    ../../src/ovps_set.cpp
    PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME ovps_set_test COMMAND ovps_set_test.x)


add_executable(ovps_test.x 
  ovps_test.cu
  ../test_helper.h
  ../../src/basis/dummy_movec_parser.cpp
  ../../src/basis/dummy_movec_parser.h
  ../../src/basis/movec_parser.cpp
  ../../src/basis/movec_parser.h
  ../../src/basis/wavefunction.cpp
  ../../src/basis/wavefunction.h
  ../../src/ovps_set.cpp
  ../../src/ovps_set.h
  ../../src/qc_mpi.cpp
  ../../src/qc_mpi.h
  ../../src/qc_ovps.cpp
  ../../src/qc_ovps.h
  ../../src/qc_random.cpp
  ../../src/qc_random.h
)
target_link_libraries(ovps_test.x gtest_main)
target_link_libraries(ovps_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_test.x tau)
target_link_libraries(ovps_test.x blas_wrapper)
target_include_directories(ovps_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(ovps_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(ovps_test.x PRIVATE ../../src/tau_generators)
target_include_directories(ovps_test.x PRIVATE ../../src/geometry)
target_compile_definitions(ovps_test.x PRIVATE -DHAVE_CUDA)
if (${Enable_CUDA})
  set_target_properties(ovps_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(ovps_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(ovps_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(
    ../../src/ovps_set.cpp
    ../../src/qc_ovps.cpp
    ../../src/basis/wavefunction.cpp
    PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME ovps_test COMMAND ovps_test.x)

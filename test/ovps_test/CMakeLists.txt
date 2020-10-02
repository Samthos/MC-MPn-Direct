set(ovps_set_test_source
  ovps_set_test.cpp
  ../../src/ovps_set.cpp
)
add_executable(ovps_set_test.x ${ovps_set_test_source})
target_link_libraries(ovps_set_test.x gtest_main)
target_link_libraries(ovps_set_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_set_test.x blas_wrapper)
target_link_libraries(ovps_set_test.x test_helper)
target_include_directories(ovps_set_test.x PRIVATE ../test_helper)
target_include_directories(ovps_set_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(ovps_set_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
if (${Enable_CUDA})
  set_target_properties(ovps_set_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(ovps_set_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(ovps_set_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(${ovps_set_test_source} PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME ovps_set_test COMMAND ovps_set_test.x)


set(ovps_test_source
  ovps_test.cpp
  ../../src/basis/dummy_movec_parser.cpp
  ../../src/basis/movec_parser.cpp
  ../../src/basis/wavefunction.cpp
  ../../src/ovps_set.cpp
  ../../src/qc_mpi.cpp
  ../../src/qc_ovps.cpp
  ../../src/qc_random.cpp
)
add_executable(ovps_test.x ${ovps_test_source})
target_link_libraries(ovps_test.x gtest_main)
target_link_libraries(ovps_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_test.x tau)
target_link_libraries(ovps_test.x blas_wrapper)
target_link_libraries(ovps_test.x test_helper)
target_include_directories(ovps_test.x PRIVATE ../test_helper)
target_include_directories(ovps_test.x PRIVATE ../../src/blas_wrapper)
target_include_directories(ovps_test.x PRIVATE ${BLAS_INCLUDE_DIRS})
target_include_directories(ovps_test.x PRIVATE ../../src/tau_generators)
target_include_directories(ovps_test.x PRIVATE ../../src/geometry)
target_compile_definitions(ovps_test.x PRIVATE -DHAVE_CUDA)
if (${Enable_CUDA})
  set_target_properties(ovps_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(ovps_test.x ${CUBLAS_LIBRARIES})
  target_compile_definitions(ovps_test.x PRIVATE -DHAVE_CUDA)
  set_source_files_properties(${ovps_test_source} PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME ovps_test COMMAND ovps_test.x)

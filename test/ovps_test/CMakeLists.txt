message("ovps test " ${gtest_SOURCE_DIR}/)

cuda_add_executable(ovps_set_test.x 
	ovps_set_test.cu
 	../../src/ovps_set.cu
	../../src/ovps_set.h
	../../src/blas_calls.h
	../../src/blas_calls.cpp
)
target_link_libraries(ovps_set_test.x gtest_main)
target_link_libraries(ovps_set_test.x ${BLAS_LIBRARIES})
target_include_directories(ovps_set_test.x PUBLIC ${BLAS_INCLUDE_DIRS})
target_compile_definitions(ovps_set_test.x PUBLIC -DHAVE_CUDA)
cuda_add_cublas_to_target(ovps_set_test.x)
add_test(NAME OVPS_SET_TEST COMMAND ovps_set_test.x)


cuda_add_executable(ovps_test.x 
	ovps_test.cu
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
target_compile_definitions(ovps_test.x PUBLIC -DHAVE_CUDA)

target_link_libraries(ovps_test.x gtest_main)
target_link_libraries(ovps_test.x ${BLAS_LIBRARIES})
target_link_libraries(ovps_test.x tau)
cuda_add_cublas_to_target(ovps_test.x)

target_include_directories(ovps_test.x PUBLIC ${BLAS_INCLUDE_DIRS})
target_include_directories(ovps_test.x PUBLIC ../../src/tau_generators)

add_test(NAME OVPS_TEST COMMAND ovps_test.x)

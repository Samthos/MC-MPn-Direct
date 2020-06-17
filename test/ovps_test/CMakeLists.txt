message("ovps test " ${gtest_SOURCE_DIR}/)

cuda_add_executable(ovps_set_test.x 
	ovps_set_test.cu
 	../../src/ovps_set_device.cu
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

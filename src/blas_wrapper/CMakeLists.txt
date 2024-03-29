set(source
  accumulate.cpp
  blas_calls.cpp
  blas_wrapper.cpp
  dasum.cpp
  daxpy.cpp
  dcopy.cpp
  ddgmm.cpp
  ddot.cpp
  ddot_batched.cpp
  dgeam.cpp
  dgekm.cpp
  dgekv.cpp
  dgemm.cpp
  dgemv.cpp
  dherk.cpp
  dscal.cpp
  dsyrk.cpp
  mfill.cpp
  shift.cpp
  transpose.cpp
  vfill.cpp
)

add_library(blas_wrapper STATIC ${source})
target_link_libraries(blas_wrapper ${BLAS_LIBRARIES})
target_include_directories(blas_wrapper PUBLIC ${BLAS_INCLUDE_DIRS})

if(${Enable_CUDA})
  set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)

  target_link_libraries(blas_wrapper ${CUBLAS_LIBRARIES})
  target_compile_definitions(blas_wrapper PRIVATE -DHAVE_CUDA)
endif()


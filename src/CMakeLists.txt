set(source
  MCF12/F12_Traces.cpp
  MCF12/F12_Traces.h
  MCF12/correlation_factor.cpp
  MCF12/correlation_factor_data.cpp
  MCF12/correlation_factor_function.cpp
  MCF12/correlation_factor_types.cpp
  MCF12/gf2_f12.cpp
  MCF12/gf2_f12.h
  MCF12/mp2_f12_v.cpp
  MCF12/mp2_f12_v.h
  MCF12/mp2_f12_vbx.cpp
  MCF12/mp2_f12_vbx.h
  MCF12/x_traces.cpp
  MCF12/x_traces.h
  MCGF/gf3_core_c.h
  MCGF/gf3_core_m_1.h
  MCGF/gf3_core_m_12.h
  MCGF/gf3_core_m_2.h
  MCGF/gf3_core_p_1.h
  MCGF/gf3_core_p_12.h
  MCGF/gf3_core_p_2.h
  MCGF/gf_full_arrays.cpp
  MCGF/gf_full_arrays.h
  MCGF/gf_stats.cpp
  MCGF/qc_gfFullStatistics.cpp
  MCGF/qc_gfFullStatisticsHost.cpp
  MCGF/qc_mcgf.cpp
  MCGF/qc_mcgf.h
  MCGF/qc_mcgf2.cpp
  MCGF/qc_mcgf2.h
  MCGF/qc_mcgf3.cpp 
  MCGF/qc_mcgf3.h
  MCMP/create_mp2_functional.cpp
  MCMP/create_mp2_functional.h
  MCMP/direct_mp2_functional.cpp
  MCMP/direct_mp2_functional.h
  MCMP/mcmp.cpp
  MCMP/mcmp.h
  MCMP/mp2_functional.cpp
  MCMP/mp2_functional.h
  MCMP/mp3_functional.cpp
  MCMP/mp3_functional.h
  MCMP/mp4_functional.cpp
  MCMP/mp4_functional.h
  MCMP/mp_functional.h
  basis/atomic_orbital.cpp
  basis/atomic_orbital.h
  basis/basis.cpp
  basis/basis.h
  basis/basis_parser.cpp
  basis/basis_parser.h
  basis/cartesian_poly.h
  basis/create_movec_parser.cpp
  basis/dummy_movec_parser.cpp
  basis/dummy_movec_parser.h
  basis/movec_parser.cpp
  basis/movec_parser.h
  basis/nwchem_movec_parser.cpp
  basis/nwchem_movec_parser.h
  basis/shell.cpp
  basis/shell.h
  basis/wavefunction.cpp
  basis/wavefunction.h
  control_variate.cpp
  control_variate.h
  main.cpp
  ovps_set.cpp
  ovps_set.h
  qc_input.cpp
  qc_monte.cpp
  qc_monte.h
  qc_mpi.cpp
  qc_mpi.h
  qc_ovps.cpp 
  qc_ovps.h
  qc_random.cpp
  qc_random.h
  timer.h
)

add_subdirectory(blas_wrapper)
add_subdirectory(geometry)
add_subdirectory(electron_generators)
add_subdirectory(tau_generators)

add_executable(MC_MPn_Direct ${source})

target_link_libraries(MC_MPn_Direct ${ARMADILLO_LIBRARIES})
target_link_libraries(MC_MPn_Direct ${BLAS_LIBRARIES})
target_link_libraries(MC_MPn_Direct blas_wrapper)
target_link_libraries(MC_MPn_Direct geometry)
target_link_libraries(MC_MPn_Direct electron_generators)
target_link_libraries(MC_MPn_Direct tau)

target_include_directories(MC_MPn_Direct PUBLIC ${ARMADILLO_INCLUDE_DIRS})
target_include_directories(MC_MPn_Direct PUBLIC ${BLAS_INCLUDE_DIRS})
target_include_directories(MC_MPn_Direct PUBLIC ./blas_wrapper)
target_include_directories(MC_MPn_Direct PUBLIC ./geometry)
target_include_directories(MC_MPn_Direct PUBLIC ./tau_generators)
target_include_directories(MC_MPn_Direct PUBLIC ./electron_generators)

IF(${MPI_CXX_FOUND})
  message("Adding MPI compile options to MC_MPn_Direct")
  target_compile_definitions(MC_MPn_Direct PUBLIC -DHAVE_MPI)
  target_link_libraries(MC_MPn_Direct ${MPI_CXX_LIBRARIES})
  target_include_directories(MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

if(${Enable_CUDA})
  set(GPU_SPECIFIC_SOURCES
    MCMP/device_mp2_functional.cu
  )
  set_source_files_properties(
      MCF12/F12_Traces.cpp
      MCF12/correlation_factor_data.cpp
      MCF12/correlation_factor_function.cpp
      MCF12/mp2_f12_v.cpp
      MCF12/mp2_f12_vbx.cpp
      MCMP/mp3_functional.cpp
      MCMP/mp4_functional.cpp
      basis/atomic_orbital.cpp
      basis/basis.cpp
      basis/wavefunction.cpp
      ovps_set.cpp
      qc_monte.cpp
      qc_ovps.cpp
      PROPERTIES LANGUAGE CUDA)


  add_executable(GPU_MC_MPn_Direct ${source} ${GPU_SPECIFIC_SOURCES})
  set_target_properties(GPU_MC_MPn_Direct PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(GPU_MC_MPn_Direct PRIVATE -DHAVE_CUDA)

  target_link_libraries(GPU_MC_MPn_Direct ${ARMADILLO_LIBRARIES})
  target_link_libraries(GPU_MC_MPn_Direct ${BLAS_LIBRARIES})
  target_link_libraries(GPU_MC_MPn_Direct ${CUBLAS_LIBRARIES})
  target_link_libraries(GPU_MC_MPn_Direct blas_wrapper)
  target_link_libraries(GPU_MC_MPn_Direct geometry)
  target_link_libraries(GPU_MC_MPn_Direct electron_generators)
  target_link_libraries(GPU_MC_MPn_Direct tau)

  target_include_directories(GPU_MC_MPn_Direct PUBLIC ${ARMADILLO_INCLUDE_DIRS})
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ${BLAS_INCLUDE_DIRS})
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./blas_wrapper)
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./geometry)
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./electron_generators)
  target_include_directories(GPU_MC_MPn_Direct PUBLIC ./tau_generators)

  IF(${MPI_CXX_FOUND})
    message("Adding MPI compile options to GPU_MC_MPn_Direct")
    target_compile_definitions(GPU_MC_MPn_Direct PUBLIC -DHAVE_MPI)
    target_link_libraries(GPU_MC_MPn_Direct ${MPI_CXX_LIBRARIES})
    target_include_directories(GPU_MC_MPn_Direct PRIVATE ${MPI_CXX_INCLUDE_PATH})
  endif()
endif()
